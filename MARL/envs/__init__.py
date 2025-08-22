from functools import partial
from .multiagentenv import MultiAgentEnv
from .traffic.TrafficGenEnvironment import TrafficGenEnvironment


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

REGISTRY["trafficgen"] = partial(env_fn, env=TrafficGenEnvironment)
