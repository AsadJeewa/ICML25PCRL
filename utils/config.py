import torch
import numpy as np

class Config:
    def __init__(self) -> None:
        self.MO_algo_name = "PreCo"
        self.mode = "train"
        self.seed = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = 0.99
        self.num_eval_weights = 100
        self.total_timesteps = 1000000

class Config_OffPolicy(Config):
    def __init__(self) -> None:
        super().__init__()
        self.learning_rate = 3e-4
        self.actor_lr = None
        self.batch_size = 256
        self.buffer_size = int(1e6)
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.05
        self.epsilon_decay_steps = 100000
        self.initial_homotopy_lambda = 0.0
        self.final_homotopy_lambda = 1.0
        self.homotopy_decay_steps = 500000
        self.learning_starts = 1000
        self.gradient_updates = 1
        self.target_net_update_freq = 1000
        self.net_arch = [256, 256, 256, 256]
        self.num_eval_episodes_for_front = 5
        self.eval_freq = 10000
        self.max_grad_norm = 1.0
        self.tau = 1
        self.warmup_steps = 150000
        self.max_episode_steps = None
        self.per = True
        self.per_alpha = 0.6
        self.initial_lam = 10.0
        self.initial_exp = 8.0
        self.lam_cap = 25.0
        self.lam_step = 0.02
        self.exp_cap = 15.0
        self.exp_step = 0.2
        self.anneal_every = 5000
        self.actor_warmup_steps = 0

class Config_OnPolicy(Config):
    def __init__(self) -> None:
        super().__init__()
        self.algo_name = "PPO"
        self.max_grad_norm = None
        self.k_epochs = 3
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.eps_clip = 0.01
        self.entropy_coef = 0.001
        self.update_freq = 100
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.train_eps = 20
        self.max_steps = 100
        self.eval_eps = 5
        self.eval_per_episode = 10
        self.test_start = 10000
        self.test_interval = 10000
        self.test_eps = 10
        self.probscale = 4
        self.net_arch = [256, 256, 256, 256]

class Config_minecart_OffPolicy(Config_OffPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = "minecart-v0"
        self.ref_point = np.array([-1.0, -1.0, -200.0])
        self.r_dim = 3
        self.num_eval_weights = 1000
        self.eval_freq = 1000
        self.gamma = 0.98
        self.learning_rate = 2e-4
        self.actor_lr = 1e-5
        self.batch_size = 32
        self.net_arch = [256, 256, 256, 256]
        self.buffer_size = int(1.5e6)
        self.initial_epsilon = 0.8
        self.final_epsilon = 0.5
        self.epsilon_decay_steps = 15000
        self.initial_homotopy_lambda = 0.9
        self.final_homotopy_lambda = 0.85
        self.homotopy_decay_steps = 10000
        self.gradient_updates = 5
        self.tau = 0.1
        self.initial_lam = 2.0
        self.initial_exp = 4.0

class Config_minecart_OnPolicy(Config_OnPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = "minecart-v0"
        self.ref_point = np.array([-1.0, -1.0, -200.0])
        self.r_dim = 3
        self.num_eval_weights = 1000
        self.test_eps = 10
        self.eval_freq = 10000

class Config_reacher_OnPolicy(Config_OnPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = "mo-reacher-v4"
        self.ref_point = np.array([-100.0, -100.0, -100.0, -100.0])
        self.r_dim = 4
        self.total_timesteps = 3000000
        self.num_eval_weights = 100
        self.train_eps = 40
        self.max_steps = 250
        self.eps_clip = 0.001
        self.entropy_coef = 0.001
        self.max_grad_norm = None
        self.test_eps = 10
        self.eval_freq = 30000

class Config_reacher_OffPolicy(Config_OffPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = "mo-reacher-v4"
        self.ref_point = np.array([-100.0, -100.0, -100.0, -100.0])
        self.r_dim = 4
        self.gamma = 0.98
        self.total_timesteps = 3000000
        self.num_eval_weights = 100
        self.eval_freq = 1000
        self.batch_size = 128
        self.buffer_size = int(5e5)
        self.learning_rate = 1e-4
        self.epsilon_decay_steps = 50000
        self.learning_starts = 5000
        self.net_arch = [256, 256, 256, 256]
        self.target_net_update_freq = 2000
        self.max_episode_steps = 250

class Config_dst_OffPolicy(Config_OffPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = "deep-sea-treasure-v0"
        self.ref_point = np.array([0.0, -50.0])
        self.r_dim = 2
        self.total_timesteps = 500000
        self.num_eval_weights = 50
        self.eval_freq = 1000
        self.net_arch = [256, 256]
        self.buffer_size = int(5e4)
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_decay_steps = 100000
        self.initial_homotopy_lambda = 0.2
        self.final_homotopy_lambda = 0.2
        self.gradient_updates = 2
        self.per = False
        self.warmup_steps = 10000
        self.initial_lam = 0.5
        self.initial_exp = 2.0
        # Lower/slower than the default exp_cap=15: a high exp makes ratio_LS in
        # compute_actor_loss collapse toward the shared ascent direction for any
        # weight not yet well-aligned with it, which can drag all weights onto
        # one policy before DST's small critic has learned per-weight values.
        self.exp_cap = 6.0
        self.exp_step = 0.05
        self.actor_warmup_steps = 20000

class Config_dst_OnPolicy(Config_OnPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = "deep-sea-treasure-v0"
        self.ref_point = np.array([0.0, -50.0])
        self.r_dim = 2
        self.total_timesteps = 500000
        self.num_eval_weights = 50
        self.train_eps = 1
        self.max_steps = 50
        self.test_eps = 5
        self.eval_freq = 10000
        self.entropy_coef = 0.01