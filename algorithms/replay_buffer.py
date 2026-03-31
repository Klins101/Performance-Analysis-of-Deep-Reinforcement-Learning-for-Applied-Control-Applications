import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr      = 0
        self.size     = 0
        self.s  = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.a  = np.zeros((max_size, action_dim), dtype=np.float32)
        self.r  = np.zeros((max_size, 1),          dtype=np.float32)
        self.s2 = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.d  = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        i = self.ptr
        self.s[i]  = state
        self.a[i]  = action
        self.r[i]  = reward
        self.s2[i] = next_state
        self.d[i]  = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (torch.as_tensor(self.s[idx]),
                torch.as_tensor(self.a[idx]),
                torch.as_tensor(self.r[idx]),
                torch.as_tensor(self.s2[idx]),
                torch.as_tensor(self.d[idx]))
