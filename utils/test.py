import numpy as np
from morl_baselines.common.weights import equally_spaced_weights

def test(cfg, env, agent):
    print("testing！")
    rewards = []
    steps = []
    mean_rs = []

    ref_vec_list = equally_spaced_weights(
        dim=cfg.r_dim,
        n=cfg.num_eval_weights,
        seed=cfg.seed + 1000
    )

    for ref_vec in ref_vec_list:
        ep_reward = np.zeros(cfg.r_dim)
        ep_step = 0

        for _ in range(cfg.test_eps):
            state = env.reset()[0]

            for _ in range(cfg.max_steps):
                ep_step += 1
                state = np.concatenate([state, ref_vec])
                action = agent.predict_action(state)
                next_state, reward, done, _, _ = env.step(action)
                state = next_state
                ep_reward += reward

                if done:
                    break

        steps.append(ep_step / cfg.test_eps)
        mean_rs.append(ep_reward / cfg.test_eps)

    print("test done")
    env.close()
    return {'rewards': mean_rs}, mean_rs, ref_vec_list