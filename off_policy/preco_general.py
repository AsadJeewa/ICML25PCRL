import argparse
import numpy as np
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from off_policy.morl_baselines.multi_policy.PCRL.PreCo import PreCo
from utils.config import Config_minecart_OffPolicy, Config_reacher_OffPolicy, Config_dst_OffPolicy
from utils.train import compute_metrics

CONFIG_REGISTRY = {
    "minecart": Config_minecart_OffPolicy,
    "reacher": Config_reacher_OffPolicy,
    "dst": Config_dst_OffPolicy,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--env", default="reacher", choices=CONFIG_REGISTRY.keys())
    parser.add_argument("--exp_notes", default="", type=str)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project_name", default="MORL-Baselines", type=str)
    parser.add_argument("--save_checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = CONFIG_REGISTRY[args.env]()
    if args.seed is not None:
        cfg.seed = args.seed

    def make_env():
        env = mo_gym.make(cfg.env_name)
        env = MORecordEpisodeStatistics(env, gamma=cfg.gamma)
        return env

    env = make_env()
    eval_env = make_env()

    experiment_name = f"PreCo_OffPolicy_{cfg.env_name}_{cfg.total_timesteps}_{args.exp_notes}_seed{cfg.seed}"

    agent = PreCo(
        env,
        max_grad_norm=cfg.max_grad_norm,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        net_arch=cfg.net_arch,
        buffer_size=cfg.buffer_size,
        initial_epsilon=cfg.initial_epsilon,
        final_epsilon=cfg.final_epsilon,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        initial_homotopy_lambda=cfg.initial_homotopy_lambda,
        final_homotopy_lambda=cfg.final_homotopy_lambda,
        homotopy_decay_steps=cfg.homotopy_decay_steps,
        learning_starts=cfg.learning_starts,
        envelope=True,
        gradient_updates=cfg.gradient_updates,
        target_net_update_freq=cfg.target_net_update_freq,
        tau=cfg.tau,
        seed=cfg.seed,
        log=args.use_wandb,
        project_name=args.project_name,
        experiment_name=experiment_name,
        per=cfg.per,
        per_alpha=cfg.per_alpha,
    )

    agent.train(
        total_timesteps=cfg.total_timesteps,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        ref_point=cfg.ref_point,
        num_eval_weights_for_front=cfg.num_eval_weights,
        eval_freq=cfg.eval_freq,
        reset_num_timesteps=False,
        reset_learning_starts=False,
        checkpoints=args.save_checkpoint,
        save_freq=100000,
        warmup_steps=cfg.warmup_steps,
        max_episode_steps=cfg.max_episode_steps,
    )

if __name__ == "__main__":
    main()