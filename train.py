from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

mode = 'all'
results_folder = './training_results/run_all_lr5e-6_cosine_b32'

model = Unet(
    dim = 64,
    channels = 3,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    mode = mode,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    beta_schedule = 'cosine',
)

config = dict(
    mode = mode,
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
    dataset_path = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v2/train',
    normalization = 'current',  # see normalize_to_neg_one_to_one in denoising_diffusion_pytorch.py
    results_folder = results_folder,
)

trainer = Trainer(
    diffusion,
    config['dataset_path'],
    mode = config['mode'],
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