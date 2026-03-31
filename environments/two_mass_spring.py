import numpy as np


class TwoMassSpringEnv:
    state_dim  = 9       # [z1, z2, dz1, dz2, ref, e, e_int, z1-z2, dz1-dz2]
    action_dim = 1
    max_action = 1.0
    def __init__(self, dt=0.1, T_final=20.0, reference=1.0,
                 k=0.3, m1=1.0, m2=1.0, c1=0.0, c2=0.0, max_abs_error=1.2):
        self.dt        = dt
        self.max_steps = int(T_final / dt)
        self.ref       = reference
        self.max_abs_error = max_abs_error

        self.A   = np.array([[0,      0,       1,       0     ],
                              [0,      0,       0,       1     ],
                              [-k/m1,  k/m1,   -c1/m1,  0     ],
                              [ k/m2, -k/m2,    0,      -c2/m2]])
        self.B   = np.array([0.0, 0.0, 1/m1, 0.0])
        self.Bw2 = np.array([0.0, 0.0, 0.0, 1/m2])
        self.C   = np.array([0.0, 1.0, 0.0, 0.0])   # output = z2

        self.reset()

    def reset(self):
        self.state      = np.zeros(4)
        self.e_int      = 0.0
        self.step_count = 0
        self.y          = 0.0
        return self._get_obs()

    def _get_obs(self):
        z1, z2, dz1, dz2 = self.state
        error = self.ref - z2
        return np.array([z1, z2, dz1, dz2,
                          self.ref, error, self.e_int,
                          z1 - z2, dz1 - dz2], dtype=np.float32)

    def step(self, action):
        u     = float(np.clip(np.asarray(action).flat[0], -self.max_action, self.max_action))
        x_dot = self.A @ self.state + self.B * u
        self.state += x_dot * self.dt
        self.y      = float(self.C @ self.state)

        error      = self.ref - self.y
        self.e_int += error * self.dt

        q_e, r_u = 100.0, 0.1
        reward = -(q_e * error**2 + r_u * u**2)
        reward += 10.0 * np.exp(-error**2 / 0.01)
        self.step_count += 1
        diverged = abs(error) > self.max_abs_error
        if diverged:
            reward -= 1000.0
        done = (self.step_count >= self.max_steps or diverged)
        return self._get_obs(), float(reward), done
