import argparse
from utils.config import Config_reacher as Config
from utils.train import train_reacher as train, compute_metrics
from utils.test import test
from utils.plot import plot_rewards
from utils.env import env_agent_config_reacher as env_agent_config
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1)
    parser.add_argument("--r", default=4)
    parser.add_argument("--m", default='PreCo')
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

    cfg = Config()
    cfg.MO_algo_name = args.m
    cfg.seed = int(args.seed)
    cfg.r_dim = int(args.r)
    cfg.use_wandb = args.use_wandb
    cfg.project_name = args.project_name
    cfg.save_checkpoint = args.save_checkpoint
    env, agent = env_agent_config(cfg)

    best_agent, res_dic, Hs = train(cfg, env, agent)
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