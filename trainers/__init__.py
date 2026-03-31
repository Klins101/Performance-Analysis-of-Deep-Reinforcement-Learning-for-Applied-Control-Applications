from . import ddpg, td3, sac, ppo, tdmpc2

TRAINERS = {
    "DDPG":   ddpg.train,
    "TD3":    td3.train,
    "SAC":    sac.train,
    "PPO":    ppo.train,
    "TDMPC2": tdmpc2.train,
}
