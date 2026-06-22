"""
Out-of-distribution experiment — three-way method comparison.

Extends the qualitative DPS-vs-U-Net OOD comparison from the diffusion
chapter (Fig. dps_ood_i, a synthetic UIUC-logo phantom) into a quantitative
three-way comparison that also includes CondDiff. This fills the gap flagged
in thesis Section "Out-of-distribution Experiment": "Quantitative results for
this three-way OOD comparison are not yet available and are left for future
work."

Mirrors /home/kamo/resources/slitless/python/experiments/diffusion_comparison/
(the in-distribution K=3, {inf,30,20} dB three-way comparison) but evaluated
on three synthetic non-EIS phantoms with sharp geometric structure -- a
repeat of the original "uiuc_i" phantom plus two additional morphologies
('blocks', 'rings') for a more systematic spatial-structure OOD sweep. Value
ranges are kept near the dset_v6 marginals so the *content*, not the scale,
is what is out of distribution.

Methods:
  - DPS:      single unconditional checkpoint (run_all_lr_1e-4_cosine_b32_logz),
              N=1000 ancestral steps, 10 posterior samples, shared across all
              noise levels.
  - CondDiff: per-noise-level conditional checkpoints, 250 DDIM steps, 10
              posterior samples.
  - U-Net:    per-noise-level supervised checkpoints (slitless), single
              point estimate.

Saves all reconstructions to outputs/results.npy. analyze.py reads this file.

Run:
    python experiments/ood_experiment/runner.py
"""
import os, sys, json
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, '/home/kamo/resources/slitless/python')

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization
from slitless.forward import forward_op_torch, add_noise
from slitless.plotting import uiuc_i, sincosgrid
from slitless.evaluate import net_loader, load_model_stats, predict
from slitless.data_loader import meas_transform, param_inv_transform

# ── config ────────────────────────────────────────────────────────────────────
PHANTOM_KINDS = ['uiuc_i', 'blocks', 'rings']

# (dbsnr, noise_model) — None/None = noiseless. Matches the in-distribution
# three-way comparison (slitless/experiments/diffusion_comparison) and the
# checkpoints available for CondDiff / U-Net.
CONFIGS = [(None, None), (30, 'gaussian'), (20, 'gaussian')]

NUMDETECTORS = 3
COND_ORDERS  = [0, -1, 1]

N_SAMPLES_DPS      = 10
N_SAMPLES_CONDDIFF = 10
DPS_STEPS          = 1000     # ancestral (ddim disabled when == timesteps)
CONDDIFF_STEPS     = 250      # ddim
GRAD_SCALE         = [0.5, 0.5, 0.5]
MILESTONE          = 10

DPS_RUN = 'training_results/run_all_lr_1e-4_cosine_b32_logz'

CONDDIFF_RUNS = {
    (None, None):     'training_results/run_all_lr1e-4_cosine_b32_conditional_logz',
    (30, 'gaussian'): 'training_results/2026_06_02__21_03_55_all_lr_1e-4_cosine_b32_numdetectors_3_global_logz_conditional_Gaussian_30',
    (20, 'gaussian'): 'training_results/2026_05_31__04_12_50_all_lr_1e-4_cosine_b32_numdetectors_3_global_logz_conditional_Gaussian_20',
}

SLITLESS_SAVED = '/home/kamo/resources/slitless/python/results/saved'
UNET_RUNS = {
    (None, None):     '2026_05_28__00_09_08_diffusion_unet_NF_64_BS_32_LR_0.0002_EP_50_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_100_None_K_3_dset_v6_logzscale',
    (30, 'gaussian'): '2026_05_31__12_00_32_diffusion_unet_NF_64_BS_32_LR_0.0002_EP_100_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_30_gaussian_K_3_dset_v6_logzscale',
    (20, 'gaussian'): '2026_05_31__13_07_15_diffusion_unet_NF_64_BS_32_LR_0.0002_EP_100_KSIZE_(3, 1)_MSE_LOSS_ADAM_all_dbsnr_20_gaussian_K_3_dset_v6_logzscale',
}

OUT_DIR = os.path.join(REPO_ROOT, 'experiments/ood_experiment/outputs')
SEED    = 0
# ─────────────────────────────────────────────────────────────────────────────

SPEEDOFLIGHT     = 299792.458
WAVELENGTH       = 195.117937907451
W_FAC            = SPEEDOFLIGHT / WAVELENGTH         # width Å -> km/s
DISPERSION_SCALE = 0.022275                          # Å/pixel
VEL_TO_PIX       = WAVELENGTH / SPEEDOFLIGHT / DISPERSION_SCALE  # km/s -> pixels
WIDTH_TO_PIX     = 1.0 / DISPERSION_SCALE            # Å -> pixels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)


def cfg_key(dbsnr, noise_model):
    return f'dbsnr_{dbsnr}_{noise_model}'


# ── synthetic OOD phantoms ──────────────────────────────────────────────────
def make_phantom(kind):
    """Synthetic non-EIS phantom with sharp geometric structure.

    Returns param_raw (3,H,W) float32: [int (erg/cm2/s/sr), vel (km/s), width (Å)].
    Value ranges chosen near the dset_v6 marginals (so the *content*, not the
    scale, is what's out-of-distribution).
    """
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]

    if kind == 'uiuc_i':
        # Bright "I" on a high-frequency sin-cos grid background -- the
        # synthetic OOD phantom from the original DPS-vs-U-Net comparison
        # (Fig. dps_ood_i), remapped to dset_v6-like physical ranges.
        mask = uiuc_i()                       # {0, 0.502, 1.0}, (64,64)
        grid = sincosgrid(64, 9, 9)           # [-1,1], (64,64)
        pattern = grid * 0.5 + mask
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())  # -> [0,1]

        inten = 300 + pattern * (3000 - 300)
        vel   = -25 + pattern * 50
        width = 0.022 + pattern * (0.035 - 0.022)
    elif kind == 'blocks':
        # 4x4 grid of piecewise-constant blocks, sharp edges
        bi, bj = yy // 16, xx // 16
        idx = (bi * 4 + bj).astype(np.float32) / 15.0   # 0..1
        inten = 300 + idx * (3000 - 300)
        vel   = -25 + idx * 50
        width = 0.022 + idx * (0.035 - 0.022)
    elif kind == 'rings':
        # concentric square rings, alternating values
        ring = np.maximum(np.abs(yy - 31.5), np.abs(xx - 31.5)).astype(np.int32)
        ring = ring % 8
        inten = np.where(ring < 4, 2500.0, 400.0)
        vel   = np.where(ring < 4, 20.0, -20.0)
        width = np.where(ring < 4, 0.033, 0.023)
    else:
        raise ValueError(kind)

    return np.stack([inten, vel, width]).astype(np.float32)   # (3,H,W)


def forward_meas(param_raw, dbsnr, noise_model='gaussian'):
    """param_raw: (N,3,H,W) [int, vel(km/s), width(Å)] -> noisy meas (N,K,H,W) DN."""
    t = torch.tensor(param_raw)
    clean = forward_op_torch(
        true_intensity=t[:, 0],
        true_doppler=t[:, 1] * VEL_TO_PIX,
        true_linewidth=t[:, 2] * WIDTH_TO_PIX,
        spectral_orders=COND_ORDERS,
    )
    noisy = add_noise(clean, dbsnr=dbsnr, noise_model=noise_model)
    return np.asarray(noisy, dtype=np.float32)


# ── model loading ────────────────────────────────────────────────────────────
def load_unet_module(run_folder, milestone, cond_channels, norm_mode):
    normalization = make_normalization(norm_mode, rec_mode='all')
    model = Unet(
        channels=3,
        cond_channels=cond_channels,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True,
    ).to(device)

    ckpt = torch.load(os.path.join(REPO_ROOT, run_folder, f'model-{milestone}.pt'),
                       map_location=device, weights_only=True)
    state = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
    model.load_state_dict(state)
    model.eval()
    return model, normalization


def build_conddiff_diffusion(run_folder, milestone):
    with open(os.path.join(REPO_ROOT, run_folder, 'config.json')) as f:
        run_cfg = json.load(f)
    model, normalization = load_unet_module(run_folder, milestone, NUMDETECTORS, run_cfg['norm_mode'])

    diffusion = GaussianDiffusion(
        model,
        mode='all',
        image_size=64,
        timesteps=1000,
        sampling_timesteps=CONDDIFF_STEPS,
        beta_schedule='cosine',
        clip_denoised=(-5., 5.),
        device=device,
        normalization=normalization,
    )
    return diffusion


def reconstruct_conddiff(diffusion, meas_np, n_samples):
    """meas_np: (K,H,W) raw DN -> samples (n_samples,3,H,W) physical units, width in km/s."""
    meas_t = torch.tensor(meas_np).to(device)[None]            # (1,K,H,W)
    cond = meas_t.expand(n_samples, -1, -1, -1)                 # (n_samples,K,H,W)
    with torch.no_grad():
        samples = diffusion.sample(batch_size=n_samples, cond=cond)
    samples = samples.cpu().numpy()
    samples[:, 2] *= W_FAC   # width Å -> km/s
    return samples


# ── DPS ──────────────────────────────────────────────────────────────────────
def forward_op_dps(x, device=None):
    """x: (N,3,H,W) physical units [int, vel(km/s), width(Å)] -> meas (N,K,H,W)."""
    return forward_op_torch(
        true_intensity=x[:, 0],
        true_doppler=x[:, 1] * VEL_TO_PIX,
        true_linewidth=x[:, 2] * WIDTH_TO_PIX,
        spectral_orders=COND_ORDERS,
        device=device,
    )


def reconstruct_dps(model, normalization, meas_np, n_samples):
    """meas_np: (K,H,W) raw DN -> samples (n_samples,3,H,W) physical units, width in km/s."""
    measurement = torch.tensor(meas_np).to(device)              # (K,H,W)

    diffusion = GaussianDiffusion(
        model,
        mode='all',
        image_size=64,
        timesteps=1000,
        sampling_timesteps=DPS_STEPS,
        recon=True,
        measurement=measurement,
        grad_scale=torch.tensor(GRAD_SCALE, dtype=torch.float32, device=device),
        forward_op=forward_op_dps,
        beta_schedule='cosine',
        clip_denoised=(-5., 5.),
        device=device,
        normalization=normalization,
    )

    samples, *_ = diffusion.sample(batch_size=n_samples)
    samples = samples.detach().cpu().numpy()
    samples[:, 2] *= W_FAC   # width Å -> km/s
    return samples


# ── U-Net ────────────────────────────────────────────────────────────────────
def load_unet_solver(run_name):
    foldpath = os.path.join(SLITLESS_SAVED, run_name)
    stats = load_model_stats(foldpath)
    net = net_loader(foldpath)
    net.eval()
    return net, stats


def reconstruct_unet(net, stats, meas_np):
    """meas_np: (K,H,W) raw DN -> recon (3,H,W) physical units, width in km/s."""
    meas_norm = meas_transform(meas_np, stats=stats, mode='log_zscore')
    recon_norm = predict(net, meas_norm)                        # (3,H,W)
    recon = param_inv_transform(recon_norm, w_kms=True, stats=stats, mode='log_zscore')
    return recon.astype(np.float32)


# ── main ──────────────────────────────────────────────────────────────────────
print('Loading DPS (unconditional) model...')
with open(os.path.join(REPO_ROOT, DPS_RUN, 'config.json')) as f:
    dps_cfg = json.load(f)
dps_model, dps_normalization = load_unet_module(DPS_RUN, MILESTONE, 0, dps_cfg['norm_mode'])

print('Loading CondDiff models...')
conddiff_diffusions = {cfg: build_conddiff_diffusion(run, MILESTONE) for cfg, run in CONDDIFF_RUNS.items()}

print('Loading U-Net models...')
unet_solvers = {cfg: load_unet_solver(run) for cfg, run in UNET_RUNS.items()}

results = {
    'config': {
        'phantom_kinds':       PHANTOM_KINDS,
        'configs':             CONFIGS,
        'numdetectors':        NUMDETECTORS,
        'cond_orders':         COND_ORDERS,
        'n_samples_dps':       N_SAMPLES_DPS,
        'n_samples_conddiff':  N_SAMPLES_CONDDIFF,
        'dps_steps':           DPS_STEPS,
        'conddiff_steps':      CONDDIFF_STEPS,
        'grad_scale':          GRAD_SCALE,
        'milestone':           MILESTONE,
        'dps_run':             DPS_RUN,
        'conddiff_runs':       CONDDIFF_RUNS,
        'unet_runs':           UNET_RUNS,
        'seed':                SEED,
    },
    'phantoms': {},
}

for kind in PHANTOM_KINDS:
    print(f'\n── phantom: {kind} ──')
    param_raw = make_phantom(kind)            # (3,H,W), width in Å
    true_kms = param_raw.copy()
    true_kms[2] *= W_FAC                       # width Å -> km/s

    phantom_results = {'true': true_kms, 'configs': {}}

    for dbsnr, noise_model in CONFIGS:
        key = cfg_key(dbsnr, noise_model)
        meas_np = forward_meas(param_raw[None], dbsnr, noise_model)[0]   # (K,H,W)

        print(f'  {key}: DPS...')
        dps_samples = reconstruct_dps(dps_model, dps_normalization, meas_np, N_SAMPLES_DPS)

        print(f'  {key}: CondDiff...')
        cond_samples = reconstruct_conddiff(conddiff_diffusions[(dbsnr, noise_model)], meas_np, N_SAMPLES_CONDDIFF)

        print(f'  {key}: U-Net...')
        net, stats = unet_solvers[(dbsnr, noise_model)]
        unet_recon = reconstruct_unet(net, stats, meas_np)

        phantom_results['configs'][key] = {
            'meas':         meas_np,          # (K,H,W)
            'dps_samples':  dps_samples,       # (N_dps,3,H,W)
            'cond_samples': cond_samples,      # (N_cond,3,H,W)
            'unet':         unet_recon,        # (3,H,W)
        }

    results['phantoms'][kind] = phantom_results

out_path = os.path.join(OUT_DIR, 'results.npy')
np.save(out_path, results)
print(f'\nSaved -> {out_path}')
