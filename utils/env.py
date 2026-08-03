import mo_gymnasium as mo_gym
from agent.agent import Agent
from utils.seed import all_seed

def env_agent_config(cfg):
    env_kwargs = {}
    if hasattr(cfg, "depth") and cfg.depth is not None:
        env_kwargs["depth"] = cfg.depth

    env = mo_gym.make(cfg.env_name, **env_kwargs)
    print("seed:", cfg.seed)
    all_seed(env, seed=cfg.seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"state dim: {n_states}，action dim: {n_actions}, reward dim: {cfg.r_dim}")
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions)
    agent = Agent(cfg)
    return env, agent 

def env_agent_config_reacher(cfg):
    env = mo_gym.make(cfg.env_name) 
    print("seed:", cfg.seed)
    all_seed(env,seed=cfg.seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"state dim: {n_states}，action dim: {n_actions}, reward dim: {cfg.r_dim}")
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions)
    agent = Agent(cfg)
    return env,agent