import numpy as np


class NMPEnv:
    state_dim  = 3       # [e_int, x1, x2]
    action_dim = 1
    max_action = 5.0

    def __init__(self, dt=0.1, T_final=20.0, reference=1.0,
                 Q=10.0, R=1.0, max_abs_eint=5.0, max_abs_error=1.5):
        self.dt        = dt
        self.max_steps = int(T_final / dt)
        self.ref       = reference
        self.Q         = Q
        self.R         = R
        self.max_abs_eint  = max_abs_eint
        self.max_abs_error = max_abs_error

        self.Ap = np.array([[0.0, 1.0], [-5.47, -4.719]])
        self.Bp = np.array([0.0, 1.0])
        self.Cp = np.array([3.199, -1.135])

        self.reset()

    def reset(self):
        self.x          = np.zeros(2)
        self.e_int      = 0.0
        self.step_count = 0
        self.y          = 0.0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.e_int, self.x[0], self.x[1]], dtype=np.float32)

    def step(self, action):
        u = float(np.clip(np.asarray(action).flat[0], -self.max_action, self.max_action))
        self.y   = float(np.dot(self.Cp, self.x))
        e        = self.ref - self.y
        self.e_int = float(np.clip(self.e_int + self.dt * e,
                                   -self.max_abs_eint, self.max_abs_eint))
        self.x  += self.dt * (self.Ap @ self.x + self.Bp * u)
        reward   = -(self.Q * self.e_int**2 + self.R * u**2)
        self.step_count += 1
        diverged = abs(e) > self.max_abs_error
        done = (self.step_count >= self.max_steps)
        return self._get_obs(), float(reward), done
