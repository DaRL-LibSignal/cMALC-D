import os
from Environment.cityflow_env import CityFlowEnv
from Environment.wrapper.default_wrapper import DefaultWrapper
from Environment.wrapper.oracle_wrapper import OracleWrapper
from Environment.wrapper.flatten_wrapper import FlattenWrapper
from Environment.wrapper.observation_wrapper import ObsWrapper


def make_env(config, wrapper_names=["DefaultWrapper"]):
    log_folder = config["log_folder"]
    work_folder = config["work_folder"]
    cityflow_config = config["cityflow_config"]
    seed = config["seed"]
    task_params = config["task_params"]
    env_type = config["env_type"]
    curriculum = config["curriculum_name"]
    test_index = config["test_index"] if "test_index" in config else -1
    env = CityFlowEnv(
        log_folder,
        work_folder,
        cityflow_config,
        task_params,
        env_type,
        curriculum,
        seed=seed,
        test_index=test_index,
    )
    for wrapper_name in wrapper_names:
        env = eval(wrapper_name)(env)
    return env
