import torch
import numpy as np
from utils.test import test
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from morl_baselines.common.performance_indicators import hypervolume, sparsity, expected_utility
import copy
import wandb
import os 

def compute_metrics(mean_rs, refs, ref_point):
    """
    Compute evaluation metrics from preference-conditioned returns.

    mean_rs:
        Achieved return vector for each evaluated preference.
        Kept in the original maximisation convention.

    refs:
        Preference vector corresponding to each return.

    ref_point:
        Hypervolume reference point in the original
        maximisation convention.
    """

    returns = np.asarray(mean_rs, dtype=np.float64)
    refs = np.asarray(refs, dtype=np.float64)

    # ---------------------------------------------------------
    # Preference alignment
    # ---------------------------------------------------------

    return_norms = np.linalg.norm(returns, axis=1, keepdims=True)
    return_norms = np.maximum(return_norms, 1e-12)

    normalised_returns = returns / return_norms

    preference_alignment = np.diag(
        normalised_returns @ refs.T
    )

    alignment = preference_alignment.mean()

    # ---------------------------------------------------------
    # Pareto front
    # ---------------------------------------------------------
    # pymoo's NonDominatedSorting assumes minimisation,
    # so negate returns ONLY for identifying the front.

    obj_values = -returns

    non_dom_idx = NonDominatedSorting().do(
        obj_values,
        only_non_dominated_front=True
    )

    # Keep front in original maximisation convention.
    pareto_front = returns[non_dom_idx]

    # ---------------------------------------------------------
    # Hypervolume
    # ---------------------------------------------------------
    # MORL-Baselines hypervolume() internally negates both
    # ref_point and points before passing them to pymoo.

    hv = hypervolume(
        ref_point=ref_point,
        points=pareto_front
    )

    # ---------------------------------------------------------
    # Sparsity + cardinality
    # ---------------------------------------------------------

    sp = sparsity(pareto_front)
    cardinality = len(pareto_front)
    nr = cardinality / len(returns)

    # ---------------------------------------------------------
    # Expected Utility
    # ---------------------------------------------------------
    # EUM expects maximisation-convention returns.

    eum = expected_utility(
        front=pareto_front,
        weights_set=refs
    )

    return {
        "HV": hv,
        "EUM": eum,
        "Sparsity": sp,
        "Cardinality": cardinality,
        "Non_Dominated_Ratio": nr,
        "Alignment": alignment,
        "pareto_indices": non_dom_idx,
    }

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0)) 
    
def train(cfg, env, agent):
    print("Starting ... ")
    rewards = []  
    steps = []
    best_ep_reward = 0 
    output_agent = None
    ref_vec_list = []
    RS = []
    Hr_l = []
    Hv = []
    NRs = []
    SPs = []
    EUMs = []
    Cardinals = []
    global_step = 0
    best_hv = -np.inf

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name,
            name=f"{cfg.MO_algo_name}_{cfg.env_name}_seed{cfg.seed}",
            config=vars(cfg)
        )

    global_step = 0
    while global_step < cfg.total_timesteps:
        print(global_step)
        if global_step%cfg.test_interval==0:
            if global_step>cfg.test_start:
                res_dic, mean_rs, refs = test(cfg, env, agent)

                ref_point = cfg.ref_point

                RS.append(np.array(mean_rs))

                metrics = compute_metrics(
                    mean_rs,
                    refs,
                    ref_point
                )

                v = metrics["HV"]
                eum = metrics["EUM"]
                sp = metrics["Sparsity"]
                cardinality = metrics["Cardinality"]
                nr = metrics["Non_Dominated_Ratio"]
                alignment = metrics["Alignment"]

                Hv.append(v)
                SPs.append(sp)
                NRs.append(nr)
                EUMs.append(eum)
                Cardinals.append(cardinality)
                Hr_l.append((global_step, alignment))

                print(
                    "HV:", v,
                    "EUM:", eum,
                    "Sparsity:", sp,
                    "Cardinality:", cardinality,
                    "ND Ratio:", nr,
                    "Alignment:", alignment
                )

                if cfg.save_checkpoint:
                    if v > best_hv:
                        best_hv = v

                        agent.save(
                            save_dir="checkpoints",
                            filename=f"best_{cfg.m}_{cfg.env}_seed{cfg.seed}"
                        )

                        if cfg.use_wandb:
                            wandb.log({
                                "best/HV": best_hv
                            })

                if cfg.use_wandb:
                # W&B evaluation logging
                    wandb.log({
                        "eval/HV": v,
                        "eval/EUM": eum,
                        "eval/alignment": alignment,
                        "eval/Sparsity": sp,
                        "eval/Cardinality": cardinality,
                        "eval/Non_Dominated_Ratio": nr,
                        "training/global_step": global_step,
                    },
                    step=global_step)

                print(cfg.seed,"seed",Hr_l)
#         ref_vec = np.zeros(cfg.r_dim)
#         ref_vec[np.random.randint(cfg.r_dim)] = 1
        
        # 2-D:
        # ref_ang = np.random.rand(cfg.r_dim-1)*np.pi/2
        # ref_vec = np.array([np.sin(ref_ang),np.cos(ref_ang)]).reshape(-1)
        
        # Higher-D
        while True:
            ref_vec = np.random.multivariate_normal(np.zeros(cfg.r_dim),np.eye(cfg.r_dim))
            if all(ref_vec>0):
                break
        ref_vec /= np.linalg.norm(ref_vec, 2)
       
        ref_vec_list.append(ref_vec)
        for i_ep in range(cfg.train_eps):
            ep_reward = 0  
            ep_step = 0
            state = env.reset()[0]  
            for _ in range(cfg.max_steps):
                if global_step >= cfg.total_timesteps:
                    break
                global_step += 1
                ep_step += 1
                state = np.concatenate([state,ref_vec])
                action = agent.sample_action(state)  
                next_state, reward, done, _ , _= env.step(action)  
                #reward = reward[1:]
                reward = reward[:cfg.r_dim]
                agent.memory.push((state, action,agent.log_probs,reward,done)) 
                state = next_state  
                agent.update(ref_vec) 
                ep_reward += reward 
                if done:
                    break
            if cfg.use_wandb:
                wandb.log({
                    "train/episode_reward_sum": np.sum(ep_reward),
                    "train/episode_length": ep_step,
                    "global_step": global_step,
                    "train/ref_0": ref_vec[0],
                    "train/ref_1": ref_vec[1] if cfg.r_dim > 1 else 0,
                    "train/ref_2": ref_vec[2] if cfg.r_dim > 2 else 0,
                },
                step=global_step
                )

            if (i_ep+1)%cfg.eval_per_episode == 0:
                sum_eval_reward = 0
                for _ in range(cfg.eval_eps):
                    eval_ep_reward = 0
                    state = env.reset()[0]
                    for _ in range(cfg.max_steps):
                        state = np.concatenate([state,ref_vec])
                        action = agent.predict_action(state) 
                        next_state, reward, done, _ , _ = env.step(action) 
                        #reward = reward[1:]
                        reward = reward[:cfg.r_dim]
                        state = next_state  
                        eval_ep_reward += reward  
                        if done:
                            break
                    sum_eval_reward += eval_ep_reward
                mean_eval_reward = sum_eval_reward/cfg.eval_eps

                if cfg.use_wandb:
                    wandb.log({
                        "eval/mean_episode_reward": np.sum(mean_eval_reward),
                        "eval/objective_0": mean_eval_reward[0],
                        "eval/objective_1": mean_eval_reward[1] if cfg.r_dim > 1 else 0,
                        "eval/objective_2": mean_eval_reward[2] if cfg.r_dim > 2 else 0,
                    },
                    step=global_step)
            if global_step >= cfg.total_timesteps:
                break
            steps.append(ep_step)
            rewards.append(ep_reward)
        
        
     
    print("done!!!!!!!!")
    output_agent = copy.deepcopy(agent) # last agent
    env.close()
    if cfg.use_wandb:
        wandb.finish()
    return output_agent, \
    {
        'rewards': rewards,
        'ref_vec_list': ref_vec_list
    }, \
    {
        'Hr': Hr_l,
        'Rs': RS,
        'Hv': Hv,
        'EUM': EUMs,
        'SP': SPs,
        'Cardinality': Cardinals,
        'NR': NRs
    }

def train_reacher(cfg, env, agent):
    print("Starting ... ")
    rewards = []  
    steps = []
    best_ep_reward = 0 
    output_agent = None
    ref_vec_list = []
    RS = []
    Hr_l = []
    Hv = []
    NRs = []
    SPs = []
    EUMs = []
    Cardinals = []

    global_step = 0

    while global_step < cfg.total_timesteps:
        if global_step >= 200:
            agent.entropy_coef /= 100
        if global_step >= 300:
            agent.entropy_coef /= 10

        if global_step % cfg.test_interval == 0:
            if global_step > cfg.test_start:
                res_dic, mean_rs, refs = test(cfg, env, agent)

                ref_point = cfg.ref_point

                RS.append(np.array(mean_rs))

                metrics = compute_metrics(
                    mean_rs,
                    refs,
                    ref_point
                )

                v = metrics["HV"]
                eum = metrics["EUM"]
                sp = metrics["Sparsity"]
                cardinality = metrics["Cardinality"]
                nr = metrics["Non_Dominated_Ratio"]
                alignment = metrics["Alignment"]

                Hv.append(v)
                SPs.append(sp)
                NRs.append(nr)
                EUMs.append(eum)
                Cardinals.append(cardinality)

                Hr_l.append((global_step, alignment))

                print(
                    "HV:", v,
                    "EUM:", eum,
                    "Sparsity:", sp,
                    "Cardinality:", cardinality,
                    "ND Ratio:", nr,
                    "Alignment:", alignment
                )
                print(cfg.seed,"seed",Hr_l)
#         ref_vec = np.zeros(cfg.r_dim)
#         ref_vec[np.random.randint(cfg.r_dim)] = 1
        
        # 2-D:
        # ref_ang = np.random.rand(cfg.r_dim-1)*np.pi/2
        # ref_vec = np.array([np.sin(ref_ang),np.cos(ref_ang)]).reshape(-1)
        
        # Higher-D
        while True:
            ref_vec = np.random.multivariate_normal(np.zeros(cfg.r_dim),np.eye(cfg.r_dim))
            if all(ref_vec>0):
                break
        ref_vec /= np.linalg.norm(ref_vec, 2)
       
        ref_vec_list.append(ref_vec)
        for i_ep in range(cfg.train_eps): # run each pref vec for cfg.train_eps episodes
            ep_reward = 0  
            ep_step = 0
            state = env.reset()[0]  
          
            for t_ in range(cfg.max_steps):
                if global_step >= cfg.total_timesteps:
                    break
                ep_step += 1
                state = np.concatenate([state,ref_vec])
                action = agent.sample_action(state)  
                next_state, reward, done, _ , _= env.step(action)  
                #reward = reward[1:]
                reward = reward[:cfg.r_dim]
                if t_ == cfg.max_steps-1:
                    done = True
                agent.memory.push((state, action,agent.log_probs,reward,done)) 
                state = next_state  
                agent.update(ref_vec) 
                ep_reward += reward 
                
                if done:
                    
                    break
            if (i_ep+1)%cfg.eval_per_episode == 0:
                sum_eval_reward = 0
                for _ in range(1):
                    eval_ep_reward = 0
                    state = env.reset()[0]
                    for _ in range(cfg.max_steps):
                        state = np.concatenate([state,ref_vec])
                        action = agent.greedy_action(state) 
                        next_state, reward, done, _ , _ = env.step(action) 
                        #reward = reward[1:]
                        reward = reward[:cfg.r_dim]
                        state = next_state  
                        eval_ep_reward += reward  
                        if done:
                            break
                        
                    sum_eval_reward += eval_ep_reward
                mean_eval_reward = sum_eval_reward
                print(mean_eval_reward,ref_vec)
                
            steps.append(ep_step)
            rewards.append(ep_reward)
        
            if global_step >= cfg.total_timesteps:
                break
     
    print("done!!!!!!!!")
    output_agent = copy.deepcopy(agent) # last agent
    env.close()
    return output_agent, \
    {
        'rewards': rewards,
        'ref_vec_list': ref_vec_list
    }, \
    {
        'Hr': Hr_l,
        'Rs': RS,
        'Hv': Hv,
        'EUM': EUMs,
        'SP': SPs,
        'Cardinality': Cardinals,
        'NR': NRs
    }
