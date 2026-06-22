"""
Uncertainty quantification experiment — reconstruction sweep.

Runs a deep ensemble of T2=4 independently-trained conditional diffusion
models (same config, different seeds) to produce T1 posterior samples per
model for three sub-experiments:

  exp1 — in-distribution: AU/EU decomposition + calibration on dset_v6 val set
  exp2 — OOD via measurement-noise-level shift (dbsnr != training dbsnr=20)
  exp3 — OOD via synthetic non-EIS phantoms (sharp geometric structure)

Saves all raw samples to outputs/results.npy. analyze.py reads this file.

Run:
    python experiments/uncertainty_quantification/runner.py
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

# ── config ────────────────────────────────────────────────────────────────────
ENSEMBLE_RUNS = [
    'training_results/2026_05_31__04_12_50_all_lr_1e-4_cosine_b32_numdetectors_3_global_logz_conditional_Gaussian_20',
    'training_results/2026_05_31__22_40_33_all_lr_1e-4_cosine_b32_numdetectors_3_global_logz_conditional_Gaussian_20',
    'training_results/2026_06_01__22_07_38_all_lr_1e-4_cosine_b32_numdetectors_3_global_logz_conditional_Gaussian_20',
    'training_results/2026_06_01__22_10_15_all_lr_1e-4_cosine_b32_numdetectors_3_global_logz_conditional_Gaussian_20',
]
MILESTONE      = 10             # final checkpoint (50k steps) for all ensemble members
T1             = 10             # posterior samples per model (exp1, exp3)
T1_EXP2        = 10             # posterior samples per model (exp2)
NUMDETECTORS   = 3              # orders [0, -1, 1]
SAMPLING_STEPS = 250
TRAIN_DBSNR    = 20             # training noise level (Gaussian)

VAL_FILE       = '/home/kamo/resources/slitless/data/datasets/baseline/eis_val_10_dsetv6.npy'
N_VAL_EXP2     = 5               # subset of val images used for the noise-shift sweep
DBSNR_LEVELS   = [30, 20, 10, 5]  # exp2: 20 = training condition (sanity check)

OUT_DIR        = os.path.join(REPO_ROOT, 'experiments/uncertainty_quantification/outputs')
# ─────────────────────────────────────────────────────────────────────────────

SPEEDOFLIGHT     = 299792.458
WAVELENGTH       = 195.117937907451
W_FAC            = SPEEDOFLIGHT / WAVELENGTH         # width Å -> km/s
DISPERSION_SCALE = 0.022275                          # Å/pixel
VEL_TO_PIX       = WAVELENGTH / SPEEDOFLIGHT / DISPERSION_SCALE  # km/s -> pixels
WIDTH_TO_PIX     = 1.0 / DISPERSION_SCALE            # Å -> pixels
ALL_ORDERS       = [0, -1, 1, -2, 2]
COND_ORDERS      = ALL_ORDERS[:NUMDETECTORS]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT_DIR, exist_ok=True)


# ── data / forward model helpers ────────────────────────────────────────────
def load_val_data(val_file):
    """Returns:
      true_kms  : (N,3,H,W) float32 — int (erg/cm2/s/sr), vel (km/s), width (km/s)
      param_raw : (N,3,H,W) float32 — int, vel (km/s), width (Å)  [for forward_op_torch]
      meas      : (N,K,H,W) float32 — raw DN, dbsnr=20 (as generated for training)
    """
    d = np.load(val_file, allow_pickle=True).item()
    param_raw = d['param3d'].astype(np.float32)         # (N,3,H,W), width in Å
    true_kms = param_raw.copy()
    true_kms[:, 2] *= W_FAC                              # width Å -> km/s
    meas = d['meas'][:, :NUMDETECTORS].astype(np.float32)
    return true_kms, param_raw, meas


def forward_meas(param_raw, dbsnr, noise_model='Gaussian'):
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


def make_phantom(kind):
    """Synthetic non-EIS phantom with sharp geometric structure.

    Returns param_raw (3,H,W) float32: [int (erg/cm2/s/sr), vel (km/s), width (Å)].
    Value ranges chosen near the dset_v6 marginals (so the *content*, not the
    scale, is what's out-of-distribution).
    """
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]

    if kind == 'blocks':
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


# ── model loading ────────────────────────────────────────────────────────────
def build_diffusion(run_folder, milestone):
    with open(os.path.join(REPO_ROOT, run_folder, 'config.json')) as f:
        run_cfg = json.load(f)
    norm_mode = run_cfg['norm_mode']

    normalization = make_normalization(norm_mode, rec_mode='all')
    model = Unet(
        channels=3,
        cond_channels=NUMDETECTORS,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True,
    ).to(device)

    ckpt = torch.load(os.path.join(REPO_ROOT, run_folder, f'model-{milestone}.pt'),
                       map_location=device, weights_only=True)
    state = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
    model.load_state_dict(state)
    model.eval()

    diffusion = GaussianDiffusion(
        model,
        mode='all',
        image_size=64,
        timesteps=1000,
        sampling_timesteps=SAMPLING_STEPS,
        beta_schedule='cosine',
        clip_denoised=(-5., 5.),
        device=device,
        normalization=normalization,
    )
    return diffusion


def reconstruct(diffusion, meas_np, n_samples):
    """meas_np: (1,K,H,W) raw DN -> samples (n_samples,3,H,W) physical units, width in km/s."""
    meas_t = torch.tensor(meas_np).to(device)
    cond = meas_t.expand(n_samples, -1, -1, -1)
    with torch.inference_mode():
        samples = diffusion.sample(batch_size=n_samples, cond=cond)
    samples = samples.cpu().numpy()
    samples[:, 2] *= W_FAC   # width Å -> km/s
    return samples


# ── main ──────────────────────────────────────────────────────────────────────
print(f'Building {len(ENSEMBLE_RUNS)} ensemble members (milestone {MILESTONE})...')
ensemble = [build_diffusion(rf, MILESTONE) for rf in ENSEMBLE_RUNS]
T2 = len(ensemble)

results = {
    'config': {
        'ensemble_runs':  ENSEMBLE_RUNS,
        'milestone':      MILESTONE,
        'T1':             T1,
        'T1_exp2':        T1_EXP2,
        'T2':             T2,
        'numdetectors':   NUMDETECTORS,
        'cond_orders':    COND_ORDERS,
        'sampling_steps': SAMPLING_STEPS,
        'train_dbsnr':    TRAIN_DBSNR,
        'val_file':       VAL_FILE,
        'n_val_exp2':     N_VAL_EXP2,
        'dbsnr_levels':   DBSNR_LEVELS,
    },
}

# ── Exp 1: in-distribution AU/EU decomposition ───────────────────────────────
print('\n── Exp 1: in-distribution ──')
true_kms, param_raw, meas_all = load_val_data(VAL_FILE)
N_VAL = len(true_kms)

samples1 = np.full((N_VAL, T2, T1, 3, 64, 64), np.nan, dtype=np.float32)
for vi in tqdm(range(N_VAL), desc='exp1'):
    meas_np = meas_all[vi:vi + 1]
    for mi, diffusion in enumerate(ensemble):
        samples1[vi, mi] = reconstruct(diffusion, meas_np, T1)

results['exp1'] = {
    'true': true_kms,        # (N_VAL,3,H,W)
    'meas': meas_all,        # (N_VAL,K,H,W)
    'samples': samples1,     # (N_VAL,T2,T1,3,H,W)
}

# ── Exp 2: OOD via measurement-noise-level shift ─────────────────────────────
print('\n── Exp 2: noise-level shift ──')
true_kms2 = true_kms[:N_VAL_EXP2]
param_raw2 = param_raw[:N_VAL_EXP2]

meas2 = np.stack([forward_meas(param_raw2, dbsnr) for dbsnr in DBSNR_LEVELS])  # (n_dbsnr, N_VAL2, K, H, W)
samples2 = np.full((len(DBSNR_LEVELS), N_VAL_EXP2, T2, T1_EXP2, 3, 64, 64), np.nan, dtype=np.float32)

for di, dbsnr in enumerate(DBSNR_LEVELS):
    for vi in tqdm(range(N_VAL_EXP2), desc=f'exp2 dbsnr={dbsnr}'):
        meas_np = meas2[di, vi:vi + 1]
        for mi, diffusion in enumerate(ensemble):
            samples2[di, vi, mi] = reconstruct(diffusion, meas_np, T1_EXP2)

results['exp2'] = {
    'true': true_kms2,       # (N_VAL2,3,H,W)
    'meas': meas2,           # (n_dbsnr,N_VAL2,K,H,W)
    'samples': samples2,     # (n_dbsnr,N_VAL2,T2,T1,3,H,W)
}

# ── Exp 3: OOD via synthetic phantom ─────────────────────────────────────────
print('\n── Exp 3: synthetic phantom ──')
PHANTOM_KINDS = ['blocks', 'rings']
phantom_true = []
phantom_meas = []
samples3 = np.full((len(PHANTOM_KINDS), T2, T1, 3, 64, 64), np.nan, dtype=np.float32)

for pi, kind in enumerate(PHANTOM_KINDS):
    p_raw = make_phantom(kind)                          # (3,H,W), width in Å
    p_kms = p_raw.copy(); p_kms[2] *= W_FAC              # width -> km/s
    p_meas = forward_meas(p_raw[None], TRAIN_DBSNR)[0]   # (K,H,W)
    phantom_true.append(p_kms)
    phantom_meas.append(p_meas)

    for mi, diffusion in enumerate(tqdm(ensemble, desc=f'exp3 {kind}')):
        samples3[pi, mi] = reconstruct(diffusion, p_meas[None], T1)

results['exp3'] = {
    'kinds': PHANTOM_KINDS,
    'true': np.stack(phantom_true),     # (n_phantom,3,H,W)
    'meas': np.stack(phantom_meas),     # (n_phantom,K,H,W)
    'samples': samples3,                # (n_phantom,T2,T1,3,H,W)
}

out_path = os.path.join(OUT_DIR, 'results.npy')
np.save(out_path, results)
print(f'\nSaved -> {out_path}')
