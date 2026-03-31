import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class _Encoder(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.LayerNorm(64), nn.ELU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim),
        )
    def forward(self, x):
        return self.net(x)


class _Dynamics(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64), nn.LayerNorm(64), nn.ELU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim),
        )
    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], dim=-1))   # residual


class _RewardFn(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64), nn.ELU(),
            nn.Linear(64, 1),
        )
    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class _TwinQ(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        def mlp():
            return nn.Sequential(
                nn.Linear(latent_dim + action_dim, 64), nn.ELU(),
                nn.Linear(64, 64), nn.ELU(),
                nn.Linear(64, 1),
            )
        self.q1 = mlp()
        self.q2 = mlp()
    def forward(self, z, a):
        sa = torch.cat([z, a], dim=-1)
        return self.q1(sa), self.q2(sa)


class _Policy(nn.Module):
    def __init__(self, latent_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ELU(),
            nn.Linear(64, 64),        nn.ELU(),
            nn.Linear(64, action_dim), nn.Tanh(),
        )
        self.max_action = max_action
    def forward(self, z):
        return self.max_action * self.net(z)



class _SeqBuffer:
    def __init__(self, state_dim, action_dim, max_size, horizon):
        self.max_size = max_size
        self.H   = horizon
        self.s   = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.a   = np.zeros((max_size, action_dim), dtype=np.float32)
        self.r   = np.zeros(max_size,               dtype=np.float32)
        self.s2  = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.d   = np.zeros(max_size,               dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.s2[i] = s2; self.d[i] = d
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_sequence(self, batch_size):
        valid = self.size - 2 * self.H
        if valid < 1:
            return None
        idxs = np.random.randint(0, valid, size=batch_size)
        obs = np.stack([self.s [(idxs + h) % self.max_size] for h in range(self.H)])
        act = np.stack([self.a [(idxs + h) % self.max_size] for h in range(self.H)])
        rew = np.stack([self.r [(idxs + h) % self.max_size] for h in range(self.H)])
        nxt = np.stack([self.s2[(idxs + h) % self.max_size] for h in range(self.H)])
        don = np.stack([self.d [(idxs + h) % self.max_size] for h in range(self.H)])
        t = lambda x: torch.as_tensor(x, dtype=torch.float32)
        return t(obs), t(act), t(rew[..., None]), t(nxt), t(don[..., None])



class TDMPC2:
    def __init__(self, state_dim, action_dim, max_action,
                 latent_dim=64, horizon=5, num_samples=256, num_pi_trajs=24,
                 mppi_temp=0.5, gamma=0.99, tau=0.005, lr=3e-4):
        self.action_dim   = action_dim
        self.max_action   = max_action
        self.latent_dim   = latent_dim
        self.horizon      = horizon
        self.num_samples  = num_samples
        self.num_pi_trajs = num_pi_trajs
        self.mppi_temp    = mppi_temp
        self.gamma        = gamma
        self.tau          = tau

        self.encoder   = _Encoder(state_dim, latent_dim)
        self.dynamics  = _Dynamics(latent_dim, action_dim)
        self.reward_fn = _RewardFn(latent_dim, action_dim)
        self.Qs        = _TwinQ(latent_dim, action_dim)
        self.policy    = _Policy(latent_dim, action_dim, max_action)

        self.encoder_target = copy.deepcopy(self.encoder)
        self.Qs_target      = copy.deepcopy(self.Qs)
        self.policy_target  = copy.deepcopy(self.policy)

        model_params = (list(self.encoder.parameters())  +
                        list(self.dynamics.parameters())  +
                        list(self.reward_fn.parameters()) +
                        list(self.Qs.parameters()))
        self.model_opt  = optim.Adam(model_params,         lr=lr)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = _SeqBuffer(state_dim, action_dim, int(1e6), horizon)

  
    def select_action(self, state):
        return self._plan(state, explore=True)

    def select_action_eval(self, state):
        with torch.no_grad():
            z = self.encoder(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            return self.policy(z).numpy()[0]

    def _plan(self, state, explore=False):
        with torch.no_grad():
            z = self.encoder(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))

            # Random action sequences
            rand_acts = (torch.rand(self.num_samples, self.horizon, self.action_dim) * 2 - 1) * self.max_action

            # Policy-seeded trajectories
            z_pi = z.expand(self.num_pi_trajs, -1).clone()
            pi_acts = []
            for h in range(self.horizon):
                a_h = self.policy(z_pi)
                pi_acts.append(a_h)
                z_pi = self.dynamics(z_pi, a_h)
            pi_acts = torch.stack(pi_acts, dim=1)            # (num_pi_trajs, H, action_dim)

            all_acts = torch.cat([rand_acts, pi_acts], dim=0)   # (N, H, action_dim)
            N = all_acts.shape[0]

            # Rollout in latent space
            z_all  = z.expand(N, -1).clone()
            values = torch.zeros(N)
            for h in range(self.horizon):
                a_h     = all_acts[:, h]
                r_h     = self.reward_fn(z_all, a_h).squeeze(-1)
                values += (self.gamma ** h) * r_h
                z_all   = self.dynamics(z_all, a_h)

            # Terminal value
            a_term    = self.policy_target(z_all)
            q1, q2    = self.Qs_target(z_all, a_term)
            values   += (self.gamma ** self.horizon) * torch.min(q1, q2).squeeze(-1)

            # MPPI weights
            scale   = self.mppi_temp * (values.std() + 1e-6)
            weights = torch.softmax(values / scale, dim=0).unsqueeze(-1).unsqueeze(-1)
            action  = (weights * all_acts).sum(dim=0)[0].clamp(-self.max_action, self.max_action)

        a = action.numpy()
        if explore:
            a = a + 0.3 * self.max_action * np.random.randn(self.action_dim)
            a = np.clip(a, -self.max_action, self.max_action)
        return a

    def reset_noise(self):
        pass


    def update(self, batch_size):
        seq = self.buffer.sample_sequence(batch_size)
        if seq is None:
            return {}

        obs, act, rew, nxt, don = seq   # each (H, batch, dim)

        # Target encodings for consistency + Q-target
        with torch.no_grad():
            z_nxt_targets = [self.encoder_target(nxt[h]) for h in range(self.horizon)]
            z_next_q      = self.encoder_target(nxt[0])
            a_next        = self.policy_target(z_next_q)
            q1_t, q2_t    = self.Qs_target(z_next_q, a_next)
            target_Q      = rew[0] + self.gamma * (1 - don[0]) * torch.min(q1_t, q2_t)

        # Encode first observation
        z    = self.encoder(obs[0])
        z0   = z                        

        consistency_loss = torch.tensor(0.0)
        reward_loss      = torch.tensor(0.0)

        for h in range(self.horizon):
            r_pred = self.reward_fn(z, act[h])
            reward_loss += F.mse_loss(r_pred, rew[h])
            z      = self.dynamics(z, act[h])
            consistency_loss += F.mse_loss(z, z_nxt_targets[h].detach())

        q1, q2   = self.Qs(z0, act[0])
        q_loss   = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        model_loss = consistency_loss + reward_loss + q_loss

        self.model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.dynamics.parameters()) +
            list(self.reward_fn.parameters()) + list(self.Qs.parameters()), 10.0)
        self.model_opt.step()

        # Policy loss (detach encoder so its gradient only comes from model_loss)
        z0_det   = self.encoder(obs[0]).detach()
        q1_pi, _ = self.Qs(z0_det, self.policy(z0_det))
        policy_loss = -q1_pi.mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        for src, tgt in [(self.encoder,  self.encoder_target),
                         (self.Qs,       self.Qs_target),
                         (self.policy,   self.policy_target)]:
            for p, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        with torch.no_grad():
            q_mean = torch.min(q1, q2).mean().item()
        return dict(consistency = consistency_loss.item(),
                    reward      = reward_loss.item(),
                    q           = q_loss.item(),
                    policy      = policy_loss.item(),
                    q_mean      = q_mean)

    def save(self, path):
        torch.save({
            "encoder":    self.encoder.state_dict(),
            "dynamics":   self.dynamics.state_dict(),
            "reward_fn":  self.reward_fn.state_dict(),
            "Qs":         self.Qs.state_dict(),
            "policy":     self.policy.state_dict(),
            "model_opt":  self.model_opt.state_dict(),
            "policy_opt": self.policy_opt.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.dynamics.load_state_dict(ckpt["dynamics"])
        self.reward_fn.load_state_dict(ckpt["reward_fn"])
        self.Qs.load_state_dict(ckpt["Qs"])
        self.policy.load_state_dict(ckpt["policy"])
        self.model_opt.load_state_dict(ckpt["model_opt"])
        self.policy_opt.load_state_dict(ckpt["policy_opt"])
        self.encoder_target = copy.deepcopy(self.encoder)
        self.Qs_target      = copy.deepcopy(self.Qs)
        self.policy_target  = copy.deepcopy(self.policy)
