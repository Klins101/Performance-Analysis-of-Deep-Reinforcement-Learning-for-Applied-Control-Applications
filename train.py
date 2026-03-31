import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import config as cfg
from environments import NMPEnv, TwoMassSpringEnv, AUVEnv, QuadcopterEnv, RMSEnv, CrazyflieEnv
from algorithms   import DDPG, TD3, SAC, PPO, TDMPC2
from trainers     import TRAINERS

ALGORITHMS = ["DDPG", "TD3", "SAC", "PPO", "TDMPC2"]

ENVIRONMENTS = {
    #"TwoMassSpring": TwoMassSpringEnv,
    #"Crazyflie":     CrazyflieEnv,
    #"NMP":           NMPEnv,
    #"AUV":           AUVEnv,
}

# ─────────────────────────────────────────────────────────────

ALG_CLASSES = {"DDPG": DDPG, "TD3": TD3, "SAC": SAC, "PPO": PPO, "TDMPC2": TDMPC2}
ALG_CFGS    = {"DDPG": cfg.DDPG_CFG, "TD3": cfg.TD3_CFG,
               "SAC":  cfg.SAC_CFG,  "PPO": cfg.PPO_CFG, "TDMPC2": cfg.TDMPC2_CFG}
COLORS      = {"DDPG": "tab:blue", "TD3": "tab:red",
               "SAC":  "tab:green", "PPO": "tab:orange", "TDMPC2": "tab:purple"}


def make_env(EnvClass):
    return EnvClass(dt=cfg.DT, T_final=cfg.T_FINAL, reference=cfg.REFERENCE)


def make_agent(alg_name, env):
    return ALG_CLASSES[alg_name](env.state_dim, env.action_dim, env.max_action,
                                 **ALG_CFGS[alg_name])


def run_episode(agent, env):
    state = env.reset().astype(np.float32)
    ts, ys, us = [], [], []
    for t in range(env.max_steps):
        action = agent.select_action_eval(state)
        state, _, _ = env.step(action)
        state = state.astype(np.float32)
        ts.append(t * env.dt)
        ys.append(env.y)
        us.append(float(action[0]))
    return np.array(ts), np.array(ys), np.array(us)


def plot_seed(env_name, seed, results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ts_ref = next(iter(results.values()))[0]
    ax1.plot(ts_ref, [cfg.REFERENCE] * len(ts_ref), "k--", lw=1.5, label="Reference")
    for alg, (ts, ys, us) in results.items():
        ax1.plot(ts, ys, label=alg, color=COLORS[alg])
        ax2.plot(ts, us, label=alg, color=COLORS[alg])
    ax1.set_ylabel("Output  y");  ax1.legend(fontsize=8)
    ax1.set_title(f"{env_name}  –  Seed {seed}")
    ax2.set_ylabel("Control  u"); ax2.set_xlabel("Time  (s)"); ax2.legend(fontsize=8)
    plt.tight_layout()
    out_dir = f"plots/{env_name}"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/seed_{seed}.png", dpi=150)
    plt.close()
    print(f"  → saved: {out_dir}/seed_{seed}.png")


if __name__ == "__main__":
    for env_name, EnvClass in ENVIRONMENTS.items():
        for seed in cfg.SEEDS:
            print(f"\n{'='*52}\n  {env_name}  |  Seed {seed}\n{'='*52}")
            np.random.seed(seed)
            torch.manual_seed(seed)

            results = {}
            for alg_name in ALGORITHMS:
                print(f"\n  [{alg_name}]")
                env   = make_env(EnvClass)
                agent = make_agent(alg_name, env)

                save_dir  = f"models/{env_name}/{alg_name}/seed_{seed}"
                os.makedirs(save_dir, exist_ok=True)

                TRAINERS[alg_name](agent, env, save_path=f"{save_dir}/best.pth")

                agent.load(f"{save_dir}/best.pth")
                results[alg_name] = run_episode(agent, make_env(EnvClass))

            plot_seed(env_name, seed, results)
