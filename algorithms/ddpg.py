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


class _Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),                    nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class _OUNoise:
    def __init__(self, action_dim, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.x = np.zeros(self.action_dim)

    def sample(self):
        self.x += self.theta * (-self.x) + self.sigma * np.random.randn(self.action_dim)
        return self.x.copy()


class DDPG:
    def __init__(self, state_dim, action_dim, max_action,
                 gamma=0.99, tau=0.005, lr_actor=1e-4, lr_critic=1e-4,
                 noise_theta=0.15, noise_sigma=0.2):
        self.actor         = _Actor(state_dim, action_dim, max_action)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic        = _Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.buffer     = ReplayBuffer(state_dim, action_dim)
        self.noise      = _OUNoise(action_dim, noise_theta, noise_sigma)
        self.gamma      = gamma
        self.tau        = tau
        self.max_action = max_action

    def select_action(self, state):
        with torch.no_grad():
            a = self.actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        return np.clip(a + self.noise.sample(), -self.max_action, self.max_action)

    def select_action_eval(self, state):
        with torch.no_grad():
            return self.actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).numpy()[0]

    def reset_noise(self):
        self.noise.reset()

    def update(self, batch_size):
        if self.buffer.size < batch_size:
            return {}
        s, a, r, s2, d = self.buffer.sample(batch_size)
        with torch.no_grad():
            target_Q = r + self.gamma * (1 - d) * self.critic_target(s2, self.actor_target(s2))
        critic_loss = nn.MSELoss()(self.critic(s, a), target_Q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        for p, pt in zip(self.actor.parameters(),  self.actor_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        with torch.no_grad():
            q_mean = self.critic(s, self.actor(s)).mean().item()
        return dict(critic=critic_loss.item(), actor=actor_loss.item(), q_mean=q_mean)

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, weights_only=True))
        self.actor_target = copy.deepcopy(self.actor)
