import argparse
from datetime import datetime
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.normalization import make_normalization

# ── config (defaults — override any value via CLI args) ───────────────────────
mode             = 'all'
norm_mode        = 'global_logz'
numdetectors     = 0                 # 0 = unconditional; >0 = conditional with [0,-1,1,-2,2][:n]
dbsnr            = 30                # dB SNR for measurement noise; None = no noise
noise_model      = 'Gaussian'
beta_schedule    = 'cosine'
train_batch_size = 32
train_lr         = 1e-4
partno           = 1                 # which partition to train on (1..partnum)
partnum          = 1                 # leakage-free partitions; 1 = full dataset
train_num_epochs = None               # total epochs (passes over data); ignored when --steps is set
train_num_steps  = 100000              # total optimizer steps; overrides epochs (use for matched-compute sweeps)
save_every       = None              # checkpoint every N epochs; None = final only
save_every_steps = 2500              # checkpoint every N steps; overrides save_every when set
sample_every     = None                # sample grid every N epochs
sample_every_steps = 2500            # sample grid every N steps; overrides sample_every when set
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Train EIS DDPM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--partno',             type=int,   default=partno)
parser.add_argument('--partnum',            type=int,   default=partnum)
parser.add_argument('--epochs',             type=int,   default=train_num_epochs,  dest='train_num_epochs')
parser.add_argument('--steps',              type=int,   default=train_num_steps,   dest='train_num_steps')
parser.add_argument('--save-every',         type=int,   default=save_every,        dest='save_every')
parser.add_argument('--save-every-steps',   type=int,   default=save_every_steps,  dest='save_every_steps')
parser.add_argument('--sample-every',       type=int,   default=sample_every,      dest='sample_every')
parser.add_argument('--sample-every-steps', type=int,   default=sample_every_steps,dest='sample_every_steps')
parser.add_argument('--lr',                 type=float, default=train_lr,          dest='train_lr')
parser.add_argument('--dbsnr',              type=float, default=dbsnr)
parser.add_argument('--numdetectors',       type=int,   default=numdetectors)
parser.add_argument('--tag',                type=str,   default='',                help='extra suffix appended to the run folder name')
args = parser.parse_args()

# apply overrides
partno             = args.partno
partnum            = args.partnum
train_num_epochs   = args.train_num_epochs
train_num_steps    = args.train_num_steps
save_every         = args.save_every
save_every_steps   = args.save_every_steps
sample_every       = args.sample_every
sample_every_steps = args.sample_every_steps
train_lr           = args.train_lr
dbsnr              = args.dbsnr
numdetectors       = args.numdetectors

method   = 'conditional' if numdetectors > 0 else 'unconditional'
lr_str   = f"{train_lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
part_str = f'_dsize_{partno}v{partnum}' if partnum > 1 else ''
tag_str  = f'_{args.tag}' if args.tag else ''
timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
results_folder = (
    f'./training_results/{timestamp}_{mode}_lr_{lr_str}_{beta_schedule}'
    f'_b{train_batch_size}_numdetectors_{numdetectors}_{norm_mode}'
    f'_{method}_{noise_model}_{dbsnr}{part_str}{tag_str}'
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
    timesteps = 1000,
    sampling_timesteps = 250,
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
    train_num_epochs = train_num_epochs,
    train_num_steps = train_num_steps,
    ema_decay = 0.995,
    save_every = save_every,
    save_every_steps = save_every_steps,
    sample_every = sample_every,
    sample_every_steps = sample_every_steps,
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
    train_num_steps = config['train_num_steps'],
    gradient_accumulate_every = config['gradient_accumulate_every'],
    ema_decay = config['ema_decay'],
    save_every = config['save_every'],
    save_every_steps = config['save_every_steps'],
    sample_every = config['sample_every'],
    sample_every_steps = config['sample_every_steps'],
    num_samples = 8,
    amp = True,
    calculate_fid = False,
)

trainer.save_config(config)
# trainer.load(51)
trainer.train()
