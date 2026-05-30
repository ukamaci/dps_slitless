from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.normalization import make_normalization

mode = 'all'
norm_mode = 'global_logz'
cond_orders = None          # set to e.g. [0, -1, 1] for conditional training
results_folder = './training_results/run_all_lr5e-6_cosine_b32'

model = Unet(
    dim = 64,
    channels = 3,
    cond_channels = len(cond_orders) if cond_orders else 0,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

normalization = make_normalization(norm_mode, rec_mode=mode)

diffusion = GaussianDiffusion(
    model,
    mode = mode,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    beta_schedule = 'cosine',
    clip_denoised = (-5., 5.),
    normalization = normalization,
)

config = dict(
    method = 'conditional' if cond_orders else 'unconditional',
    mode = mode,
    cond_orders = cond_orders,
    image_size = 64,
    timesteps = 1000,
    sampling_timesteps = 250,
    beta_schedule = 'cosine',
    train_batch_size = 32,
    gradient_accumulate_every = 2,
    train_lr = 5e-6,
    train_num_steps = 100000,
    ema_decay = 0.995,
    save_and_sample_every = 1000,
    dataset_path = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data/train',
    norm_mode = norm_mode,
    clip_denoised = (-5., 5.),
    results_folder = results_folder,
)

trainer = Trainer(
    diffusion,
    config['dataset_path'],
    mode = config['mode'],
    cond_orders = config['cond_orders'],
    results_folder = config['results_folder'],
    train_batch_size = config['train_batch_size'],
    train_lr = config['train_lr'],
    train_num_steps = config['train_num_steps'],          # total training steps
    gradient_accumulate_every = config['gradient_accumulate_every'],    # gradient accumulation steps
    ema_decay = config['ema_decay'],                # exponential moving average decay
    save_and_sample_every = config['save_and_sample_every'],
    num_samples = 8,
    amp = True,                       # turn on mixed precision
    calculate_fid = False,             # whether to calculate fid during training
)

trainer.save_config(config)
# trainer.load(51)
trainer.train()