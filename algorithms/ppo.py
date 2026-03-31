import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),       nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.log_std    = nn.Parameter(-0.5 * torch.ones(action_dim))
        self.max_action = max_action

    def forward(self, x):
        mean = self.net(x)
        std  = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist   = torch.distributions.Normal(mean, std)
        u      = dist.rsample()
        log_prob = dist.log_prob(u).sum(-1, keepdim=True)
        action = u.clamp(-self.max_action, self.max_action)
        return action, u, log_prob, dist.entropy().sum(-1, keepdim=True)

    def evaluate(self, state, u):
        mean, std = self.forward(state)
        dist     = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(u).sum(-1, keepdim=True)
        entropy  = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy


class _Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),       nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class PPO:
    def __init__(self, state_dim, action_dim, max_action,
                 gamma=0.99, lam=0.95, lr_actor=3e-4, lr_critic=1e-3,
                 clip_eps=0.2, n_epochs=10, n_steps=2048,
                 n_minibatches=4, entropy_coef=0.0):
        self.actor      = _Actor(state_dim, action_dim, max_action)
        self.critic     = _Critic(state_dim)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma        = gamma
        self.lam          = lam
        self.clip_eps     = clip_eps
        self.n_epochs     = n_epochs
        self.n_steps      = n_steps
        self.n_minibatches = n_minibatches
        self.entropy_coef  = entropy_coef
        self.max_action    = max_action

    def select_action(self, state):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, u, log_prob, _ = self.actor.sample(s)
        return action.numpy()[0], u.numpy()[0], log_prob.item()

    def select_action_eval(self, state):
        with torch.no_grad():
            mean, _ = self.actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return mean.numpy()[0]

    def reset_noise(self):
        pass

    def update_from_rollout(self, states, us, returns, advantages, old_log_probs):
        s_t   = torch.as_tensor(np.array(states),     dtype=torch.float32)
        u_t   = torch.as_tensor(np.array(us),         dtype=torch.float32)
        r_t   = torch.as_tensor(np.array(returns),    dtype=torch.float32).unsqueeze(1)
        adv_t = torch.as_tensor(np.array(advantages), dtype=torch.float32).unsqueeze(1)
        olp_t = torch.as_tensor(np.array(old_log_probs), dtype=torch.float32).unsqueeze(1)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        n     = len(states)
        mb    = n // self.n_minibatches

        actor_losses, critic_losses = [], []
        for _ in range(self.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, mb):
                i        = idx[start: start + mb]
                lp, ent  = self.actor.evaluate(s_t[i], u_t[i])
                ratio    = (lp - olp_t[i]).exp()
                surr     = torch.min(ratio * adv_t[i],
                                     ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[i])
                a_loss   = -surr.mean() - self.entropy_coef * ent.mean()
                self.actor_opt.zero_grad();  a_loss.backward();  self.actor_opt.step()
                actor_losses.append(a_loss.item())

                c_loss   = nn.MSELoss()(self.critic(s_t[i]), r_t[i])
                self.critic_opt.zero_grad(); c_loss.backward(); self.critic_opt.step()
                critic_losses.append(c_loss.item())

        return dict(actor=float(np.mean(actor_losses)), critic=float(np.mean(critic_losses)))

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, weights_only=True))
