import argparse
from utils.config import Config_minecart, Config_reacher, Config_dst
from utils.train import train, train_reacher, compute_metrics
from utils.test import test
from utils.plot import plot_rewards
from utils.env import env_agent_config, env_agent_config_reacher
import numpy as np
import sys

CONFIG_REGISTRY = {
    "minecart": Config_minecart,
    "reacher": Config_reacher,
    "dst": Config_dst,
}

TRAIN_REGISTRY = {
    "minecart": train,
    "dst": train,
    "reacher": train_reacher,
}

ENV_REGISTRY = {
    "minecart": env_agent_config,
    "dst": env_agent_config,
    "reacher": env_agent_config_reacher,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1)
    parser.add_argument("--r", default=None, type=int)
    parser.add_argument("--m", default='PreCo')
    parser.add_argument("--env", default="minecart", choices=CONFIG_REGISTRY.keys())
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--project_name",
        default="MORL-Baselines",
        type=str,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--save_checkpoint",
        action="store_true",
        help="Save checkpoints"
    )
    args = parser.parse_args()

    cfg = CONFIG_REGISTRY[args.env]()
    cfg.MO_algo_name = args.m
    cfg.seed = int(args.seed)
    if args.r is not None:
        cfg.r_dim = args.r
    cfg.use_wandb = args.use_wandb
    cfg.project_name = args.project_name
    cfg.save_checkpoint = args.save_checkpoint
    env, agent = ENV_REGISTRY[args.env](cfg)

    best_agent, res_dic, Hs = TRAIN_REGISTRY[args.env](cfg, env, agent)
    res_dic, mean_rs, refs = test(cfg, env, best_agent)

    metrics = compute_metrics(
        mean_rs,
        refs,
        cfg.ref_point
    )

    print(
        "HV:", metrics["HV"],
        "EUM:", metrics["EUM"],
        "Sparsity:", metrics["Sparsity"],
        "Cardinality:", metrics["Cardinality"],
        "ND Ratio:", metrics["Non_Dominated_Ratio"],
        "Alignment:", metrics["Alignment"]
    )
    print(cfg.seed, "seed")

    plot_rewards(res_dic['rewards'], cfg, tag="train")
    np.set_printoptions(threshold=sys.maxsize)
    print(repr(np.array(mean_rs)), repr(np.array(refs)))

if __name__ == "__main__":
    main()