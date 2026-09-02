import torch
import numpy as np

class Config:
    def __init__(self) -> None:
        self.probscale = 4
        self.new_step_api = False
        self.algo_name = "PPO"
        self.MO_algo_name = "PreCo"
        self.mode = "train"
        self.seed = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = 0.99
        self.k_epochs = 3
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.eps_clip = 0.01
        self.entropy_coef = 0.001
        self.update_freq = 100
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256

class Config_minecart(Config):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = "minecart-v0"
        self.ref_point = np.array([-1.0, -1.0, 200.0])
        self.r_dim = 3
        self.train_eps = 20
        self.total_timesteps = 1000000
        self.test_interval = 10000
        self.test_start = 10000
        self.test_eps = 10
        self.test_res = 10
        self.num_eval_weights = 1000
        self.max_steps = 100
        self.eval_eps = 5
        self.eval_per_episode = 10

class Config_reacher(Config):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = "mo-reacher-v4"
        self.ref_point = np.array([-50.0, -50.0, -50.0, -50.0])
        self.r_dim = 4
        self.train_eps = 40
        self.total_timesteps = 1000000
        self.test_interval = 10000
        self.test_start = 10000
        self.test_eps = 5
        self.test_res = 10
        self.num_eval_weights = 100
        self.max_steps = 250
        self.eval_eps = 5
        self.eval_per_episode = 10
        self.eps_clip = 0.001
        self.entropy_coef = 0.001

class Config_dst(Config):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = "deep-sea-treasure-v0"
        self.ref_point = np.array([0.0, -50.0])
        self.r_dim = 2
        self.train_eps = 1
        self.total_timesteps = 500000
        self.test_interval = 10000
        self.test_start = 10000
        self.test_eps = 5
        self.test_res = 10
        self.num_eval_weights = 50
        self.max_steps = 50
        self.eval_eps = 5
        self.eval_per_episode = 10
        self.entropy_coef = 0.01