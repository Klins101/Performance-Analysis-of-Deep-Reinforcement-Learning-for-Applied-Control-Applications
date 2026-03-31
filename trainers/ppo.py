import os
import numpy as np
import torch
import config as cfg
from .logger import Logger


def train(agent, env, save_path):
    log         = Logger()
    state       = env.reset().astype(np.float32)
    best_ret    = -1e9
    total_steps = 0
    last_eval   = 0

    while total_steps < cfg.TOTAL_TIMESTEPS:
        states, us_raw, actions, rewards, dones, log_probs, values = [], [], [], [], [], [], []
        ep_ret = 0.0

        for _ in range(agent.n_steps):
            action, u, log_prob = agent.select_action(state)
            with torch.no_grad():
                value = agent.critic(
                    torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                ).item()

            next_state, reward, done = env.step(action)
            states.append(state);    us_raw.append(u)
            actions.append(action);  rewards.append(reward)
            dones.append(float(done)); log_probs.append(log_prob); values.append(value)
            ep_ret += reward

            state       = next_state.astype(np.float32) if not done else env.reset().astype(np.float32)
            total_steps += 1

            if done:
                diverged = env.step_count < env.max_steps
                log.log_episode(ep_ret, diverged=diverged)
                ep_ret = 0.0

        # GAE
        with torch.no_grad():
            last_val = agent.critic(
                torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ).item()
        gae, returns, advantages = 0.0, [], []
        next_v = last_val
        for i in reversed(range(len(rewards))):
            delta  = rewards[i] + agent.gamma * next_v * (1 - dones[i]) - values[i]
            gae    = delta + agent.gamma * agent.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_v = values[i]
            returns.insert(0, gae + values[i])

        losses = agent.update_from_rollout(states, us_raw, returns, advantages, log_probs)
        losses["q_mean"] = float(np.mean(values))   
        log.add_losses(losses)

        if total_steps - last_eval >= cfg.EVAL_FREQ:
            avg_ret   = _evaluate(agent, env)
            last_eval = total_steps
            log.log_eval(total_steps, avg_ret)
            print(f"    steps={total_steps:>7}  eval={avg_ret:+.1f}")
            if avg_ret > best_ret:
                best_ret = avg_ret
                agent.save(save_path)

    if not os.path.exists(save_path):
        agent.save(save_path)

    log.log_compute(agent, env)
    log.save(save_path.replace(".pth", "_log.npz"))


def _evaluate(agent, env):
    returns = []
    for _ in range(cfg.EVAL_EPISODES):
        state, done, ep_ret = env.reset().astype(np.float32), False, 0.0
        while not done:
            state, reward, done = env.step(agent.select_action_eval(state))
            state   = state.astype(np.float32)
            ep_ret += reward
        returns.append(ep_ret)
    return float(np.mean(returns))
