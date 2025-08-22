import sys
import os
from copy import deepcopy
import numpy as np
import torch
import yaml
import wandb
from Environment.utils.arguments import parse_args

import yaml
from MARL.run import run
from MARL.utils.logging import get_logger

logger = get_logger()


def _get_config(config_file):
    if config_file is not None:
        with open(
            config_file,
            "r",
        ) as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{} error: {}".format(config_file, exc)
        return config_dict
    else:
        return {}


def main(_config, _log):
    config = deepcopy(_config)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["enable_wandb"]:
        wandb.login()
        wandb.init(
            project=config["wandb_project_name"],
            entity=config["wandb_entity_name"],
            name=config["name"],
            config=config,
        )
    run(config, _log)


if __name__ == "__main__":
    args = sys.argv
    if any("--config" in arg for arg in args):
        index = -1
        for i, arg in enumerate(args):
            if "--config" in arg:
                index = i
                break
        alg_config = _get_config("MARL/configs/algs/default.yaml")
        alg_config.update(_get_config(args[index].split("=")[1]))
        name = os.path.basename(args[index].split("=")[1]).split(".")[0]
        args = args[:index] + args[index + 1 :]
    for arg in args:
        if "--cityflow-config" in arg:
            cityflow_name = arg.split("=")[1]
            cityflow_name = os.path.basename(cityflow_name).split(".")[0]
    command = args[0]
    args = vars(parse_args(args))
    args["config"] = alg_config
    args["data_path"] = "/cmlscratch/anirudhs/trafficgen/data"
    args["name"] = name
    args["cityflow_name"] = cityflow_name
    args["results_file"] = os.path.join(
        args["results_path"],
        f"{args['name']}_{args['cityflow_name']}_{args['env_type']}_{args['curriculum']}_{args['seed']}.json",
    )
    os.makedirs(args["results_path"], exist_ok=True)
    if os.path.exists(args["results_file"]) and not args["overwrite"]:
        print(f"Results file {args['results_file']} already exists. Skipping...")
        sys.exit()
    main(args, logger)
