from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

mode = 'int'

model = Unet(
    dim = 64,
    channels = 1,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    mode = mode,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v2/train',
    mode=mode,
    results_folder='./results_int',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 50000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    save_and_sample_every = 1000,
    amp = True,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

trainer.load(6)
trainer.train()