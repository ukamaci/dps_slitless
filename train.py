from datetime import datetime
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.normalization import make_normalization

# ── config ────────────────────────────────────────────────────────────────────
mode             = 'all'
norm_mode        = 'global_logz'
numdetectors     = 3                 # 0 = unconditional; >0 = conditional with [0,-1,1,-2,2][:n]
dbsnr            = 30                # dB SNR for measurement noise; None = no noise
noise_model      = 'Gaussian'
beta_schedule    = 'cosine'
train_batch_size = 32
train_lr         = 1e-4
partno           = 1                 # which partition to train on (1..partnum); dataset-size / no-leakage ablation
partnum          = 1                 # split the training set into this many leakage-free partitions; 1 = full set

method    = 'conditional' if numdetectors > 0 else 'unconditional'
lr_str    = f"{train_lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
part_str  = f'_dsize_{partno}v{partnum}' if partnum > 1 else ''
timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
results_folder = (
    f'./training_results/{timestamp}_{mode}_lr_{lr_str}_{beta_schedule}'
    f'_b{train_batch_size}_numdetectors_{numdetectors}_{norm_mode}'
    f'_{method}_{noise_model}_{dbsnr}{part_str}'
)

model = Unet(
    dim = 64,
    channels = 3,
    cond_channels = numdetectors,
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
    beta_schedule = beta_schedule,
    clip_denoised = (-5., 5.),
    normalization = normalization,
)

config = dict(
    method = method,
    mode = mode,
    numdetectors = numdetectors,
    dbsnr = dbsnr,
    noise_model = noise_model,
    partno = partno,
    partnum = partnum,
    image_size = 64,
    timesteps = 1000,
    sampling_timesteps = 250,
    beta_schedule = beta_schedule,
    train_batch_size = train_batch_size,
    gradient_accumulate_every = 2,
    train_lr = train_lr,
    train_num_epochs = 140,
    ema_decay = 0.995,
    save_every = None,          # epochs between checkpoints; None = save only the final model
    sample_every = 14,          # epochs between sample grids
    dataset_path = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data/train',
    norm_mode = norm_mode,
    clip_denoised = (-5., 5.),
    results_folder = results_folder,
)

trainer = Trainer(
    diffusion,
    config['dataset_path'],
    mode = config['mode'],
    numdetectors = config['numdetectors'],
    dbsnr = config['dbsnr'],
    noise_model = config['noise_model'],
    partno = config['partno'],
    partnum = config['partnum'],
    results_folder = config['results_folder'],
    train_batch_size = config['train_batch_size'],
    train_lr = config['train_lr'],
    train_num_epochs = config['train_num_epochs'],
    gradient_accumulate_every = config['gradient_accumulate_every'],
    ema_decay = config['ema_decay'],
    save_every = config['save_every'],
    sample_every = config['sample_every'],
    num_samples = 8,
    amp = True,
    calculate_fid = False,
)

trainer.save_config(config)
# trainer.load(51)
trainer.train()
