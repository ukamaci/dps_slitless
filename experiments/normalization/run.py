"""
Normalization experiment runner.
Trains two models — global_logz and persample_linear — on dset_v6.
Run from repo root: python experiments/normalization/run.py [logz|persample]
"""
import sys
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.normalization import make_normalization

variant = sys.argv[1] if len(sys.argv) > 1 else 'logz'
assert variant in ('logz', 'persample'), "Usage: run.py [logz|persample]"

mode      = 'all'
norm_mode = 'global_logz' if variant == 'logz' else 'persample_linear'
results_folder = f'./training_results/exp_norm_{variant}_dset6_lr5e-6'

normalization = make_normalization(norm_mode, rec_mode=mode)

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
    timesteps = 1000,
    sampling_timesteps = 250,
    beta_schedule = 'cosine',
    clip_denoised = (-5., 5.),
    normalization = normalization,
)

config = dict(
    mode = mode,
    image_size = 64,
    timesteps = 1000,
    sampling_timesteps = 250,
    beta_schedule = 'cosine',
    clip_denoised = (-5., 5.),
    train_batch_size = 32,
    gradient_accumulate_every = 2,
    train_lr = 5e-6,
    train_num_steps = 50000,
    ema_decay = 0.995,
    save_and_sample_every = 1000,
    dataset_path = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data/train',
    norm_mode = norm_mode,
    results_folder = results_folder,
)

trainer = Trainer(
    diffusion,
    config['dataset_path'],
    mode = config['mode'],
    results_folder = config['results_folder'],
    train_batch_size = config['train_batch_size'],
    train_lr = config['train_lr'],
    train_num_steps = config['train_num_steps'],
    gradient_accumulate_every = config['gradient_accumulate_every'],
    ema_decay = config['ema_decay'],
    save_and_sample_every = config['save_and_sample_every'],
    num_samples = 8,
    amp = True,
    calculate_fid = False,
)

trainer.save_config(config)
trainer.train()
