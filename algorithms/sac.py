import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from .replay_buffer import ReplayBuffer

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class _GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),       nn.ReLU(),
        )
        self.mean_l    = nn.Linear(64, action_dim)
        self.log_std_l = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, x, deterministic=False):
        feat    = self.net(x)
        mean    = self.mean_l(feat)
        log_std = self.log_std_l(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()
        u       = mean if deterministic else mean + std * torch.randn_like(mean)
        action  = torch.tanh(u) * self.max_action
        # log_prob with tanh correction
        dist    = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(u).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - (action / self.max_action).pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob


class _TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        def mlp():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 64), nn.ReLU(),
                nn.Linear(64, 64),                    nn.ReLU(),
                nn.Linear(64, 1),
            )
        self.q1 = mlp()
        self.q2 = mlp()

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


class SAC:
    def __init__(self, state_dim, action_dim, max_action,
                 gamma=0.99, tau=0.005, lr_actor=3e-4, lr_critic=3e-4,
                 lr_alpha=3e-4, init_temperature=0.2):
        self.actor         = _GaussianActor(state_dim, action_dim, max_action)
        self.critic        = _TwinCritic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.log_alpha  = torch.tensor(np.log(init_temperature), dtype=torch.float32, requires_grad=True)
        self.alpha_opt  = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = -float(action_dim)
        self.buffer     = ReplayBuffer(state_dim, action_dim)
        self.gamma      = gamma
        self.tau        = tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        with torch.no_grad():
            a, _ = self.actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return a.numpy()[0]

    def select_action_eval(self, state):
        with torch.no_grad():
            a, _ = self.actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0), deterministic=True)
        return a.numpy()[0]

    def reset_noise(self):
        pass

    def update(self, batch_size):
        if self.buffer.size < batch_size:
            return {}
        s, a, r, s2, d = self.buffer.sample(batch_size)

        with torch.no_grad():
            a2, lp2    = self.actor(s2)
            q1_t, q2_t = self.critic_target(s2, a2)
            target_Q   = r + self.gamma * (1 - d) * (torch.min(q1_t, q2_t) - self.alpha.detach() * lp2)
        q1, q2      = self.critic(s, a)
        critic_loss = nn.MSELoss()(q1, target_Q) + nn.MSELoss()(q2, target_Q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        a_new, lp   = self.actor(s)
        q1n, q2n    = self.critic(s, a_new)
        actor_loss  = (self.alpha.detach() * lp - torch.min(q1n, q2n)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss  = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        with torch.no_grad():
            a_det, _ = self.actor(s)
            q1d, q2d = self.critic(s, a_det)
            q_mean   = torch.min(q1d, q2d).mean().item()
        return dict(critic=critic_loss.item(), actor=actor_loss.item(),
                    alpha=self.alpha.item(), q_mean=q_mean)

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, weights_only=True))
