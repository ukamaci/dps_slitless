"""
Conditional diffusion normalization experiment — reconstruction sweep.

Sweeps over (run, milestone, val_image) and saves all results to
outputs/results.npy. Run this once; analyze.py reads the saved file.

Edit the config block below, then:
    python experiments/conddiff_normalization/runner.py
"""
import os, sys, json
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization

# ── config ────────────────────────────────────────────────────────────────────
RUNS = {
    'logz':   'training_results/run_all_lr1e-4_cosine_b32_conditional_logz',
    'linear': 'training_results/run_all_lr1e-4_cosine_b32_conditional_linear',
}
MILESTONES     = [1, 2, 4, 8, 10]
N_SAMPLES      = 10             # posterior samples per val image
NUMDETECTORS   = 3              # use first NUMDETECTORS orders from [0,-1,1,-2,2]
SAMPLING_STEPS = 250
VAL_FILE       = '/home/kamo/resources/slitless/data/datasets/baseline/eis_val_10_dsetv6.npy'
OUT_DIR        = os.path.join(REPO_ROOT, 'experiments/conddiff_normalization/outputs')
# ─────────────────────────────────────────────────────────────────────────────

SPEEDOFLIGHT  = 299792.458
WAVELENGTH    = 195.117937907451
W_FAC         = SPEEDOFLIGHT / WAVELENGTH           # width Å → km/s
ALL_ORDERS    = [0, -1, 1, -2, 2]
COND_ORDERS   = ALL_ORDERS[:NUMDETECTORS]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT_DIR, exist_ok=True)


def load_val_data(val_file):
    """Load validation file.

    Expects a .npy dict with:
      'param3d' : (N, 3, H, W)  [int (erg/cm²/s/sr), vel (km/s), width (Å)]
      'meas'    : (N, 5, H, W)  orders [0,-1,1,-2,2]
    Returns true (N,3,H,W) with width in km/s, and meas (N,K,H,W).
    """
    d    = np.load(val_file, allow_pickle=True).item()
    true = d['param3d'].astype(np.float32)                      # (N,3,H,W)
    true[:, 2] *= W_FAC                                         # width Å → km/s
    meas = d['meas'][:, :NUMDETECTORS].astype(np.float32)       # (N,K,H,W)
    return true, meas


def build_diffusion(run_folder, milestone, norm_mode, cond_orders):
    with open(os.path.join(REPO_ROOT, run_folder, 'config.json')) as f:
        run_cfg = json.load(f)

    normalization = make_normalization(norm_mode, rec_mode='all')
    model = Unet(
        channels     = 3,
        cond_channels= len(cond_orders),
        dim          = 64,
        dim_mults    = (1, 2, 4, 8),
        flash_attn   = True,
    ).to(device)

    ckpt  = torch.load(os.path.join(REPO_ROOT, run_folder, f'model-{milestone}.pt'),
                       map_location=device, weights_only=True)
    state = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
    model.load_state_dict(state)
    model.eval()

    diffusion = GaussianDiffusion(
        model,
        mode               = 'all',
        image_size         = 64,
        timesteps          = 1000,
        sampling_timesteps = SAMPLING_STEPS,
        beta_schedule      = 'cosine',
        clip_denoised      = (-5., 5.),
        device             = device,
        normalization      = normalization,
    )
    return diffusion, normalization


def reconstruct(diffusion, normalization, norm_mode, meas_np, n_samples):
    """Run conditional sampling for one val image.

    meas_np : (1, K, H, W) numpy float32
    Returns samples (n_samples, 3, H, W) in physical units (width in km/s).
    """
    meas_t = torch.tensor(meas_np).to(device)
    if norm_mode == 'persample_linear':
        normalization.set_infer_scale(meas_t[:, 0].max())

    cond = meas_t.expand(n_samples, -1, -1, -1)   # (n_samples, K, H, W)
    with torch.inference_mode():
        samples = diffusion.sample(batch_size=n_samples, cond=cond)
    samples = samples.cpu().numpy()                # (n_samples, 3, H, W)
    samples[:, 2] *= W_FAC                         # width Å → km/s
    return samples


def rmse_ch(a, b):
    """a, b: (..., H, W) → scalar RMSE per leading dims."""
    return np.sqrt(np.mean((a - b) ** 2, axis=(-1, -2)))


# ── main sweep ────────────────────────────────────────────────────────────────
print(f'Loading val data from {VAL_FILE}')
true_all, meas_all = load_val_data(VAL_FILE)    # (N_VAL, 3, H, W) width in km/s, (N_VAL, K, H, W)
N_VAL = len(true_all)
print(f'  {N_VAL} val images loaded')

results = {
    'config': {
        'runs':           RUNS,
        'milestones':     MILESTONES,
        'n_samples':      N_SAMPLES,
        'numdetectors':   NUMDETECTORS,
        'cond_orders':    COND_ORDERS,
        'sampling_steps': SAMPLING_STEPS,
        'val_file':       VAL_FILE,
    },
    'true': true_all,    # (N_VAL, 3, H, W)  — width in km/s
    'meas': meas_all,    # (N_VAL, K, H, W)
}

for run_name, run_folder in RUNS.items():
    with open(os.path.join(REPO_ROOT, run_folder, 'config.json')) as f:
        run_cfg = json.load(f)
    norm_mode = run_cfg['norm_mode']
    print(f'\n── {run_name}  ({norm_mode}) ──')

    # (N_MS, N_VAL, N_SAMPLES, 3, H, W) and (N_MS, N_VAL, N_SAMPLES, 3)
    all_samples   = np.full((len(MILESTONES), N_VAL, N_SAMPLES, 3, 64, 64), np.nan, dtype=np.float32)
    all_rmse_samp = np.full((len(MILESTONES), N_VAL, N_SAMPLES, 3), np.nan, dtype=np.float32)
    all_rmse_mean = np.full((len(MILESTONES), N_VAL, 3), np.nan, dtype=np.float32)

    for ms_idx, ms in enumerate(MILESTONES):
        ckpt_path = os.path.join(REPO_ROOT, run_folder, f'model-{ms}.pt')
        if not os.path.exists(ckpt_path):
            print(f'  [skip] milestone {ms} not found'); continue

        diffusion, normalization = build_diffusion(run_folder, ms, norm_mode, COND_ORDERS)

        for val_idx in tqdm(range(N_VAL), desc=f'  ms={ms}'):
            meas_np = meas_all[val_idx : val_idx + 1]   # (1, K, H, W)
            true_np = true_all[val_idx]                  # (3, H, W)

            samples = reconstruct(diffusion, normalization, norm_mode, meas_np, N_SAMPLES)
            mean_s  = samples.mean(axis=0)               # (3, H, W)

            all_samples[ms_idx, val_idx]   = samples
            all_rmse_samp[ms_idx, val_idx] = rmse_ch(true_np[None], samples)  # (N_SAMPLES, 3)
            all_rmse_mean[ms_idx, val_idx] = rmse_ch(true_np, mean_s)         # (3,)

        ms_rmse_mean = all_rmse_mean[ms_idx].mean(axis=0)  # (3,)
        print(f'  ms={ms:>2}  int={ms_rmse_mean[0]:.1f}  vel={ms_rmse_mean[1]:.2f}  width={ms_rmse_mean[2]:.2f}')

    results[run_name] = {
        'norm_mode':    norm_mode,
        'samples':      all_samples,      # (N_MS, N_VAL, N_SAMPLES, 3, H, W)
        'rmse_samples': all_rmse_samp,    # (N_MS, N_VAL, N_SAMPLES, 3)
        'rmse_mean':    all_rmse_mean,    # (N_MS, N_VAL, 3)
    }

out_path = os.path.join(OUT_DIR, 'results.npy')
np.save(out_path, results)
print(f'\nSaved → {out_path}')