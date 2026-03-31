DT       = 0.1
T_FINAL  = 20.0
REFERENCE = 1.0

SEEDS = [0, 1, 2, 3, 4]

TOTAL_TIMESTEPS  = 200_000
BATCH_SIZE       = 312
EVAL_FREQ        = 5_000
EVAL_EPISODES    = 3
START_TIMESTEPS  = 5_000   


DDPG_CFG = dict(
    gamma      = 0.99,
    tau        = 0.005,
    lr_actor   = 1e-4,
    lr_critic  = 1e-3,
    noise_theta = 0.15,
    noise_sigma = 0.2,
)

TD3_CFG = dict(
    gamma        = 0.99,
    tau          = 0.005,
    lr_actor     = 1e-4,
    lr_critic    = 1e-3,
    policy_noise = 0.2,
    noise_clip   = 0.5,
    policy_delay = 2,
)

SAC_CFG = dict(
    gamma            = 0.99,
    tau              = 0.005,
    lr_actor         = 1e-4,
    lr_critic        = 1e-4,
    lr_alpha         = 1e-4,
    init_temperature = 0.2,
)

PPO_CFG = dict(
    gamma       = 0.99,
    lam         = 0.95,
    lr_actor    = 1e-4,
    lr_critic   = 1e-3,
    clip_eps    = 0.2,
    n_epochs    = 500,
    n_steps     = 6048,
    n_minibatches = 4,
    entropy_coef  = 0.0,
)

TDMPC2_CFG = dict(
    latent_dim   = 64,
    horizon      = 8,
    num_samples  = 512,
    num_pi_trajs = 24,
    mppi_temp    = 0.5,
    gamma        = 0.99,
    tau          = 0.005,
    lr           = 1e-4,
)
