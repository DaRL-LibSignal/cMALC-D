import datetime
import glob
import os
import re
import threading
import time
import copy
from os.path import abspath, dirname
from types import SimpleNamespace as SN
import pandas as pd
import numpy as np
import torch
import pdb
from datetime import datetime
import wandb
from MARL.components.episode_buffer import ReplayBuffer
from MARL.components.transforms import OneHot
from MARL.components.reward_scaler import RewardScaler
from MARL.controllers import REGISTRY as mac_REGISTRY
from MARL.learners import REGISTRY as le_REGISTRY
from MARL.envs import REGISTRY as env_REGISTRY
from MARL.runners.episode_runner import EpisodeRunner
from MARL.utils.logging import Logger
from MARL.utils.timehelper import time_left, time_str
import json
from Environment.curriculum_generator import *
from tqdm import tqdm


class TestPhase:
    def __init__(self, runner, task_generator, env_registry, args):
        self.runner = runner
        self.task_generator = task_generator
        self.env_registry = env_registry
        self.args = args
        self.args.mode = "test"
        self.test_results = []
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.learner_path = f"{args.log_folder}/{args.curriculum_name}_{args.seed}_{args.env_type}_{args.run_type}_{curr_time}/"
        os.makedirs(self.learner_path, exist_ok=True)

        # Generate fixed set of test tasks at initialization
        self.test_tasks = [
            self.task_generator.generate_task() for _ in range(self.args.test_levels)
        ]
        self.current_test_idx = 0

    def get_next_test_task(self):
        """Get the next test task in the fixed sequence"""
        if self.current_test_idx >= len(self.test_tasks):
            return None  # All test tasks completed

        task = self.test_tasks[self.current_test_idx]
        self.current_test_idx += 1
        return task

    def run_test(self, learner, num_episodes=10):
        """Evaluate on all held-out test tasks"""
        results = []
        learner.save_models(self.learner_path)

        while True:
            task = self.get_next_test_task()
            if task is None:
                break  # All test tasks completed

            level_results = {
                "task_params": task,
                "before_finetune": self._evaluate_level(task),
                "after_finetune": None,
            }

            # Optional: Add fine-tuning if desired
            if num_episodes > 0:
                level_results["after_finetune"] = self._finetune_level(
                    learner, task, num_episodes
                )

            results.append(level_results)
        return results

    def _evaluate_level(self, task):
        """Evaluate current policy on a test level"""
        self.args.test_task = task
        self.runner.env = self.env_registry[self.args.env](
            config=vars(self.args), curriculum=None
        )
        returns, _ = self.runner.run(test_mode=True)
        return np.mean(returns).item()

    def _finetune_level(self, learner, task, num_episodes):
        """Fine-tune policy on a specific test level"""
        self.args.test_task = task
        self.runner.env = self.env_registry[self.args.env](
            config=vars(self.args), curriculum=None
        )

        fine_tune_returns = []
        learner.load_models(self.learner_path)
        for _ in range(num_episodes):
            episode_batch, train_reward, train_stats = self.runner.run(test_mode=False)
            if self.args.config["use_reward_normalization"]:
                episode_batch = RewardScaler().transform(episode_batch)
            metrics = learner.train(episode_batch, self.runner.t_env, 0)
            train_stats.update({"Train Reward": train_reward})
            test_reward, test_stats = self.runner.run(test_mode=True)
            train_stats.update({"Test Reward": test_reward})
            train_stats.update(test_stats)
            train_stats.update(metrics)
            print(train_stats)
            fine_tune_returns.append(train_stats)

        return fine_tune_returns


def run(_config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    tmp_config = {
        k: _config[k]
        for k in _config
        if type(_config[k]) != dict and type(_config[k]) != list
    }
    tmp_config.update({f"config.{k}": _config["config"][k] for k in _config["config"]})
    print(
        pd.Series(tmp_config, name="HyperParameter Value")
        .transpose()
        .sort_index()
        .fillna("")
        .to_markdown()
    )

    for key, value in args.config.items():
        setattr(args, key, value)

    args.use_cuda = torch.cuda.is_available()
    args.device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    log_dict = run_sequential(args=args, logger=logger)
    with open(f"{args.results_file}", "w") as f:
        json.dump(log_dict, f, indent=4)

    # Clean up after finishing
    print("Exiting Main")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
    os._exit(os.EX_OK)


def run_sequential(args, logger):
    # Initialize environment and curriculum
    if args.env_type == "car":
        task_generator = CarTaskGenerator(args.model_name, args.device, args.run_type)
    else:
        raise NotImplementedError

    curriculum = eval(args.curriculum)(task_generator, args.config)
    args.metrics = {}
    runner = EpisodeRunner(args=args, logger=logger, curriculum=curriculum)

    # Set up schemes and groups
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "mean_action": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int,
        },
        "probs": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "reward": {"vshape": (1,)},
        "individual_rewards": {"vshape": (1,), "group": "agents"},
        "total_reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.config["buffer_size"],
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.config["buffer_cpu_only"] else args.device,
    )

    logger.console_logger.info("MDP Components:")
    print(pd.DataFrame(buffer.scheme).transpose().sort_index().fillna("").to_markdown())

    # Setup multiagent controller
    mac = mac_REGISTRY[args.config["mac"]](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Initialize learner
    learner = le_REGISTRY[args.config["learner"]](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    # Generate held-out test tasks at the beginning
    test_task_generator = CarTaskGenerator(
        args.model_name, args.device, "random", args.seed
    )
    test_tasks = [test_task_generator.generate_task() for _ in range(5)]

    # Store test results
    test_history = []

    # Training loop
    episode = 0
    last_test_T = 0
    last_log_T = 0
    test_best_return = 0
    log_dicts = []
    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(
        "Beginning training for {} timesteps".format(args.config["t_max"])
    )

    # Pre-collect samples for reward normalization
    if args.config["use_reward_normalization"]:
        episode_batch, train_old_return, train_stats = runner.run(test_mode=False)
        reward_scaler = RewardScaler()
        reward_scaler.fit(episode_batch)

    while runner.t_env <= args.config["t_max"]:
        # Collect samples
        with torch.no_grad():
            episode_batch, train_reward, train_stats = runner.run(test_mode=False)
            wandb_dict = {
                "Train Reward": np.mean(train_reward).item(),
                **{f"train_{k}": v for k, v in train_stats.items()},
            }
            print(train_stats)

            if args.config["use_reward_normalization"]:
                episode_batch = reward_scaler.transform(episode_batch)
            buffer.insert_episode_batch(episode_batch)

        # Train
        if buffer.can_sample(args.config["batch_size"]):
            next_episode = episode + args.config["batch_size_run"]
            if (args.accumulated_episodes is None) or (
                args.accumulated_episodes
                and next_episode % args.accumulated_episodes == 0
            ):
                episode_sample = buffer.sample(args.config["batch_size"])
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if args.use_cuda and episode_sample.device == "cpu":
                    episode_sample.to("cuda")
                learner_stats = learner.train(episode_sample, runner.t_env, episode)
                wandb_dict.update(learner_stats)

        # Evaluate on held-out test tasks periodically
        if runner.t_env >= last_test_T + args.config["test_interval"]:
            logger.console_logger.info(
                f"t_env: {runner.t_env} / {args.config['t_max']}\n"
                f"Estimated time left: {time_left(last_time, last_test_T, runner.t_env, args.config['t_max'])}\n"
                f"Time passed: {time_str(time.time() - start_time)}"
            )
            last_time = time.time()
            last_test_T = runner.t_env

            # Evaluate on all held-out test tasks
            test_results = {}
            for i, task in enumerate(test_tasks):
                args.test_task = task
                args.mode = "test"
                runner.env = env_REGISTRY[args.env](
                    config=vars(args), curriculum=curriculum
                )
                returns, stats = runner.run(test_mode=True)
                test_results[f"test_task_{i}"] = {
                    "params": task,
                    "return": np.mean(returns).item(),
                    "stats": stats,
                }
                args.mode = "train"

            test_history.append({"t_env": runner.t_env, "results": test_results})
            print(test_results)

            # Log to wandb
            wandb_dict.update(
                {
                    "Test/Step": runner.t_env,
                    **{
                        f"Test/Task_{i}_Return": res["return"]
                        for i, res in test_results.items()
                    },
                }
            )

        # Log
        if args.enable_wandb:
            wandb.log(wandb_dict, step=runner.t_env)
        wandb_dict.update({"Time Step": runner.t_env, "Task": curriculum.current_task})
        log_dicts.append(wandb_dict)

        # Update curriculum
        episode += args.config["batch_size_run"]
        if (runner.t_env - last_log_T) >= args.config["log_interval"]:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

        args.metrics = {**wandb_dict, "episode": episode}
        curriculum.update_curriculum(args.metrics)
        print(wandb_dict["Task"])
        runner.env = env_REGISTRY[args.env](config=vars(args), curriculum=curriculum)

    # Final evaluation
    test_phase = TestPhase(runner, test_task_generator, env_REGISTRY, args)
    final_results = test_phase.run_test(
        learner, num_episodes=args.config.get("finetune_episodes", 5)
    )

    log_dicts.append(
        {
            "test_history": test_history,
            "final_test_results": final_results,
            "best_test_return": test_best_return,
        }
    )

    logger.console_logger.info("Finished Training")
    return log_dicts


def args_sanity_check(config, _log):
    if config["use_cuda"] and not torch.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["config"]["test_nepisode"] < config["config"]["batch_size_run"]:
        config["config"]["test_nepisode"] = config["config"]["batch_size_run"]
    else:
        config["config"]["test_nepisode"] = (
            config["config"]["test_nepisode"] // config["config"]["batch_size_run"]
        ) * config["config"]["batch_size_run"]

    return config
