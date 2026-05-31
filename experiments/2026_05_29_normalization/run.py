"""
Normalization experiment — training.

Edit the config block below, then: python experiments/normalization/run.py
"""
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.normalization import make_normalization

# ── config ────────────────────────────────────────────────────────────────────
NORM_MODE = 'global_logz'        # 'global_logz' or 'persample_linear'
# ─────────────────────────────────────────────────────────────────────────────

tag            = 'logz' if NORM_MODE == 'global_logz' else 'persample'
results_folder = f'./training_results/exp_norm_{tag}_dset6_lr5e-6'

normalization = make_normalization(NORM_MODE, rec_mode='all')

model = Unet(
    dim        = 64,
    channels   = 3,
    dim_mults  = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    mode               = 'all',
    image_size         = 64,
    timesteps          = 1000,
    sampling_timesteps = 250,
    beta_schedule      = 'cosine',
    clip_denoised      = (-5., 5.),
    normalization      = normalization,
)

config = dict(
    norm_mode              = NORM_MODE,
    mode                   = 'all',
    image_size             = 64,
    timesteps              = 1000,
    sampling_timesteps     = 250,
    beta_schedule          = 'cosine',
    clip_denoised          = (-5., 5.),
    train_batch_size       = 32,
    gradient_accumulate_every = 2,
    train_lr               = 5e-6,
    train_num_steps        = 50000,
    ema_decay              = 0.995,
    save_and_sample_every  = 1000,
    dataset_path           = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data/train',
    results_folder         = results_folder,
)

trainer = Trainer(
    diffusion,
    config['dataset_path'],
    mode                      = config['mode'],
    results_folder            = config['results_folder'],
    train_batch_size          = config['train_batch_size'],
    train_lr                  = config['train_lr'],
    train_num_steps           = config['train_num_steps'],
    gradient_accumulate_every = config['gradient_accumulate_every'],
    ema_decay                 = config['ema_decay'],
    save_and_sample_every     = config['save_and_sample_every'],
    num_samples               = 8,
    amp                       = True,
    calculate_fid             = False,
)

trainer.save_config(config)
trainer.train()