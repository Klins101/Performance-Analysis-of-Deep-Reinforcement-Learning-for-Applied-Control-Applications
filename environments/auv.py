import numpy as np


class AUVEnv:
    # obs = [error, x1, x2, x3, e_int]
    state_dim  = 5
    action_dim = 1
    max_action = 5.0

    def __init__(self, dt=0.1, T_final=20.0, reference=1.0,
                 Q=10.0, R=0.1, Q_int=1.0, max_abs_error=1.5):
        self.dt        = dt
        self.max_steps = int(T_final / dt)
        self.ref       = reference
        self.Q         = Q
        self.R         = R
        self.Q_int     = Q_int
        self.max_abs_error = max_abs_error

        self.Ap = np.array([[0, 1, 0],
                             [0, 0, 1],
                             [0, -5.375, -5.235]])
        self.Bp = np.array([0.0, 1.816, -3.770])
        self.Cp = np.array([1.0, 0.0, 0.0])   # output = x1

        self.reset()

    def reset(self):
        self.x          = np.zeros(3)
        self.e_int      = 0.0
        self.step_count = 0
        self.y          = 0.0
        return self._get_obs()

    def _get_obs(self):
        self.y = float(np.dot(self.Cp, self.x))
        error  = self.ref - self.y
        return np.array([error, self.x[0], self.x[1], self.x[2], self.e_int],
                        dtype=np.float32)

    def step(self, action):
        u = float(np.clip(np.asarray(action).flat[0], -self.max_action, self.max_action))
        self.y  = float(np.dot(self.Cp, self.x))
        error   = self.ref - self.y
        self.e_int += self.dt * error
        self.x     += self.dt * (self.Ap @ self.x + self.Bp * u)
        reward  = -(self.Q * error**2 + self.R * u**2 + self.Q_int * self.e_int**2)
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        return self._get_obs(), float(reward), done
