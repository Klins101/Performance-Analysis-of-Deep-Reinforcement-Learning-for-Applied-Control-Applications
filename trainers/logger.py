import time
import numpy as np
import torch
import os

PERF_THRESHOLD = 0.8   # fraction of best eval return used for sample-efficiency timestep


def count_params(agent):
    total = 0
    for v in vars(agent).values():
        if isinstance(v, torch.nn.Module):
            total += sum(p.numel() for p in v.parameters())
    return total


def measure_inference_ms(agent, env, n=1000):
    s = env.reset().astype(np.float32)
    for _ in range(50):
        agent.select_action_eval(s)
    t0 = time.perf_counter()
    for _ in range(n):
        agent.select_action_eval(s)
    return (time.perf_counter() - t0) / n * 1000


class Logger:
    def __init__(self):
        self.episode_returns   = []
        self.eval_steps        = []
        self.eval_returns      = []
        self._loss_buf         = {}
        self.loss_log          = {}       
        self.loss_std_log      = {}       
        self._ep_ret_buf       = []     
        self.reward_variance   = []       
        self.divergence_count  = 0        
        self.training_time_s   = None
        self.inference_ms      = None
        self.n_params          = None
        self._t_start          = time.perf_counter()

    def log_episode(self, ep_return, diverged=False):
        self.episode_returns.append(float(ep_return))
        self._ep_ret_buf.append(float(ep_return))
        if diverged:
            self.divergence_count += 1

    def add_losses(self, loss_dict):
        for k, v in loss_dict.items():
            self._loss_buf.setdefault(k, []).append(float(v))

    def log_eval(self, t, eval_return):
        self.eval_steps.append(int(t))
        self.eval_returns.append(float(eval_return))

        for k, vals in self._loss_buf.items():
            self.loss_log.setdefault(k, []).append(float(np.mean(vals)))
            self.loss_std_log.setdefault(k, []).append(float(np.std(vals)))
        self._loss_buf = {}

        self.reward_variance.append(float(np.std(self._ep_ret_buf)) if len(self._ep_ret_buf) > 1 else 0.0)
        self._ep_ret_buf = []

    def log_compute(self, agent, env):
        self.training_time_s = time.perf_counter() - self._t_start
        self.inference_ms    = measure_inference_ms(agent, env)
        self.n_params        = count_params(agent)

    def _auc(self):
        if len(self.eval_returns) < 2:
            return 0.0
        steps = np.array(self.eval_steps, dtype=float)
        rets  = np.array(self.eval_returns, dtype=float)
        return float(np.trapezoid(rets, steps))

    def _timesteps_to_threshold(self):
        if not self.eval_returns:
            return None
        best    = max(self.eval_returns)
        target  = PERF_THRESHOLD * best
        for t, r in zip(self.eval_steps, self.eval_returns):
            if r >= target:
                return int(t)
        return None

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = dict(
            episode_returns   = np.array(self.episode_returns),
            eval_steps        = np.array(self.eval_steps),
            eval_returns      = np.array(self.eval_returns),
            # stability 
            reward_variance   = np.array(self.reward_variance),
            # safety 
            divergence_count  = np.array(self.divergence_count),
            # sample efficiency
            auc               = np.array(self._auc()),
            timesteps_to_threshold = np.array(
                self._timesteps_to_threshold() if self._timesteps_to_threshold() is not None else -1
            ),
            # compute 
            training_time_s   = np.array(self.training_time_s if self.training_time_s is not None else -1),
            inference_ms      = np.array(self.inference_ms    if self.inference_ms    is not None else -1),
            n_params          = np.array(self.n_params        if self.n_params        is not None else -1),
        )
        for k, v in self.loss_log.items():
            data[f"loss_{k}"]     = np.array(v)
            data[f"loss_{k}_std"] = np.array(self.loss_std_log.get(k, []))
        np.savez(path, **data)
