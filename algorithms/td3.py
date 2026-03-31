import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from .replay_buffer import ReplayBuffer


class _Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),       nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)


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

    def q1_val(self, s, a):
        return self.q1(torch.cat([s, a], dim=-1))


class TD3:
    def __init__(self, state_dim, action_dim, max_action,
                 gamma=0.99, tau=0.005, lr_actor=3e-4, lr_critic=3e-4,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.actor         = _Actor(state_dim, action_dim, max_action)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic        = _TwinCritic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.buffer     = ReplayBuffer(state_dim, action_dim)
        self.gamma        = gamma
        self.tau          = tau
        self.max_action   = max_action
        self.policy_noise = policy_noise * max_action
        self.noise_clip   = noise_clip   * max_action
        self.policy_delay = policy_delay
        self._step        = 0

    def select_action(self, state):
        with torch.no_grad():
            a = self.actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        noise = np.random.randn(len(a)) * self.policy_noise * 0.5
        return np.clip(a + noise, -self.max_action, self.max_action)

    def select_action_eval(self, state):
        with torch.no_grad():
            return self.actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).numpy()[0]

    def reset_noise(self):
        pass

    def update(self, batch_size):
        if self.buffer.size < batch_size:
            return {}
        self._step += 1
        s, a, r, s2, d = self.buffer.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2    = (self.actor_target(s2) + noise).clamp(-self.max_action, self.max_action)
            q1_t, q2_t = self.critic_target(s2, a2)
            target_Q = r + self.gamma * (1 - d) * torch.min(q1_t, q2_t)
        q1, q2      = self.critic(s, a)
        critic_loss = nn.MSELoss()(q1, target_Q) + nn.MSELoss()(q2, target_Q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        losses = dict(critic=critic_loss.item())
        if self._step % self.policy_delay == 0:
            actor_loss = -self.critic.q1_val(s, self.actor(s)).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            for p, pt in zip(self.actor.parameters(),  self.actor_target.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            losses["actor"] = actor_loss.item()
        with torch.no_grad():
            losses["q_mean"] = self.critic.q1_val(s, self.actor(s)).mean().item()
        return losses

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, weights_only=True))
        self.actor_target = copy.deepcopy(self.actor)
