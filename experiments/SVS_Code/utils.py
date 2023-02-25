def from_env_name_to_class(env_name):
    import importlib
    module = importlib.import_module('gym_pybullet_drones.envs.multi_agent_rl.' + env_name)
    env_class = getattr(module, env_name)
    return env_class


def build_env_by_name(env_class, **kwargs):
    temp_kwargs = kwargs.copy()
    #temp_kwargs["gui"] = False  # This will avoid two spawned gui
    temp_env = env_class(**temp_kwargs)
    return lambda _: env_class(**kwargs), temp_env.observation_space, temp_env.action_space, temp_env
