from MARL.envs import REGISTRY as env_REGISTRY
from functools import partial
from MARL.components.episode_buffer import EpisodeBatch
import numpy as np
import time
import faulthandler


class EpisodeRunner:
    def __init__(self, args, logger, curriculum):

        self.args = args
        self.logger = logger
        # For EpisodeRunner set batch_size default to 1
        # self.batch_size = self.args.batch_size_run
        self.batch_size = 1
        # assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](
            config=vars(self.args), curriculum=curriculum
        )
        self.env_info = self.env.get_env_info()
        self.episode_limit = self.env_info["episode_limit"]
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_rewards = []
        self.test_rewards = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env_info

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        self.train_returns = []
        self.test_returns = []
        self.train_rewards = []
        self.test_rewards = []
        self.train_stats = {}
        self.test_stats = {}

    def compute_reward(self, env_info):
        reward = env_info["total_reward"]
        return reward

    def run(self, test_mode=False):
        self.reset()
        self.test_rewards = []
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )
            actions = actions
            reward, terminated, env_info = self.env.step(actions.cpu().numpy())
            episode_return += reward
            if test_mode:
                self.test_rewards.append(self.compute_reward(env_info))
            else:
                self.train_rewards.append(self.compute_reward(env_info))
            post_transition_data = {
                "actions": actions[0].reshape(-1).to("cpu").detach(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "individual_rewards": env_info["individual_rewards"],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch,
            t_ep=self.t,
            t_env=self.t_env,
            test_mode=test_mode,
        )
        self.batch.update(
            {"actions": actions[0].reshape(-1).to("cpu").detach()}, ts=self.t
        )

        if not test_mode:
            self.t_env += self.t
            self.train_reward = sum(self.train_rewards)
            for k in env_info:
                if k in [
                    "average_time",
                    "throughput",
                    "average_wait_time",
                    "average_delay",
                    "success_rate",
                ]:
                    self.train_stats[k] = env_info[k]
            return self.batch, self.train_reward, self.train_stats
        else:
            self.test_reward = sum(self.test_rewards)
            for k in env_info:
                if k in [
                    "average_time",
                    "throughput",
                    "average_wait_time",
                    "average_delay",
                    "success_rate",
                ]:
                    self.test_stats[k] = env_info[k]
            return self.test_reward, self.test_stats

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()
