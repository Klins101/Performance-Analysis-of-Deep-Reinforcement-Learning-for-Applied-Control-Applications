import os
import numpy as np
import config as cfg
from .logger import Logger


def train(agent, env, save_path):
    log      = Logger()
    state    = env.reset().astype(np.float32)
    best_ret = -1e9
    ep_ret   = 0.0

    for t in range(1, cfg.TOTAL_TIMESTEPS + 1):
        if t < cfg.START_TIMESTEPS:
            action = np.random.uniform(-env.max_action, env.max_action, env.action_dim)
        else:
            action = agent.select_action(state)

        next_state, reward, done = env.step(action)
        agent.buffer.add(state, action, reward, next_state.astype(np.float32), float(done))
        state   = next_state.astype(np.float32)
        ep_ret += reward

        if done:
            diverged = env.step_count < env.max_steps
            log.log_episode(ep_ret, diverged=diverged)
            ep_ret = 0.0
            state  = env.reset().astype(np.float32)

        losses = agent.update(cfg.BATCH_SIZE)
        if losses:
            log.add_losses(losses)

        if t % cfg.EVAL_FREQ == 0:
            avg_ret = _evaluate(agent, env)
            log.log_eval(t, avg_ret)
            print(f"    t={t:>7}  eval={avg_ret:+.1f}")
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
