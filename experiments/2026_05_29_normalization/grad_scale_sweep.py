"""
grad_scale sweep for normalization experiment.

Edit the config block below, then: python experiments/normalization/grad_scale_sweep.py
"""
import glob
import os
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization
from slitless.forward import forward_op_torch

# ── config ────────────────────────────────────────────────────────────────────
MILESTONE   = 10
N_TEST      = 5
DPS_SAMPLES = 5
GRAD_SCALES = [0.3, 0.5, 1.0, 2.0, 5.0]
# ─────────────────────────────────────────────────────────────────────────────

SPEEDOFLIGHT     = 299792.458
WAVELENGTH       = 195.117937907451
W_FAC            = SPEEDOFLIGHT / WAVELENGTH
DISPERSION_SCALE = 0.022275
VEL_TO_PIX       = WAVELENGTH / SPEEDOFLIGHT / DISPERSION_SCALE
WIDTH_TO_PIX     = 1.0 / DISPERSION_SCALE

DATA_DIR  = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data/test'
RUN_LOGZ  = os.path.join(REPO_ROOT, 'training_results/exp_norm_logz_dset6_lr5e-6')
RUN_PS    = os.path.join(REPO_ROOT, 'training_results/exp_norm_persample_dset6_lr5e-6')
OUT_DIR   = os.path.join(REPO_ROOT, 'experiments/normalization/outputs')

CH_LABELS = ['int (erg/cm²/s/sr)', 'vel (km/s)', 'width (km/s)']
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── helpers ───────────────────────────────────────────────────────────────────
def load_test_sample(idx):
    files = sorted(glob.glob(DATA_DIR + '/data*.npy'))
    d = np.load(files[idx], allow_pickle=True).item()
    meas_np = np.stack([d[f'meas_{o}'] for o in [0, -1, 1]])[None].astype(np.float32)
    true_np = np.stack([d['int'], d['vel'], d['width']])[None].astype(np.float32)
    return meas_np, true_np


def build_diffusion(run_folder, norm_mode, meas_np, true_np, grad_scale):
    normalization = make_normalization(norm_mode, rec_mode='all')
    if norm_mode == 'persample_linear':
        normalization.set_infer_scale(torch.tensor(meas_np[:, 0]).max())

    model = Unet(channels=3, dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(device)
    ckpt = torch.load(f'{run_folder}/model-{MILESTONE}.pt', map_location=device, weights_only=True)
    state = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
    model.load_state_dict(state)
    model.eval()

    meas_t = torch.tensor(meas_np).to(device)
    true_t = torch.tensor(true_np).to(device)

    def forward_op(x, device=None):
        return forward_op_torch(true_intensity=x[:, 0],
                                true_doppler=x[:, 1] * VEL_TO_PIX,
                                true_linewidth=x[:, 2] * WIDTH_TO_PIX, device=device)

    return GaussianDiffusion(
        model,
        mode='all',
        image_size=64,
        timesteps=1000,
        sampling_timesteps=1000,
        recon=True,
        measurement=meas_t,
        true=true_t,
        beta_schedule='cosine',
        clip_denoised=(-5., 5.),
        grad_scale=torch.tensor([grad_scale]).to(device),
        forward_op=forward_op,
        device=device,
        normalization=normalization,
    )


def run_one(run_folder, norm_mode, sample_idx, grad_scale):
    meas_np, true_np = load_test_sample(sample_idx)
    diffusion = build_diffusion(run_folder, norm_mode, meas_np, true_np, grad_scale)

    samples, *_ = diffusion.sample(batch_size=DPS_SAMPLES)
    samples = samples.detach().cpu().numpy()

    samples_ph = samples.copy(); samples_ph[:, 2] *= W_FAC
    true_ph    = true_np.copy(); true_ph[:, 2]    *= W_FAC

    mean_r = samples_ph.mean(axis=0)
    return np.sqrt(np.mean((true_ph[0] - mean_r) ** 2, axis=(-1, -2)))  # (C,)


# ── main ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

runs    = [('global_logz', RUN_LOGZ), ('persample_linear', RUN_PS)]
results = {nm: [] for nm, _ in runs}

all_jobs = [(nm, rf, gs, idx)
            for (nm, rf), gs, idx in product(runs, GRAD_SCALES, range(N_TEST))]

with tqdm(all_jobs, desc='sweep', unit='run') as pbar:
    for norm_mode, run_folder, gs, idx in pbar:
        pbar.set_postfix(norm=norm_mode[:6], gs=gs, sample=idx)
        r = run_one(run_folder, norm_mode, idx, gs)
        tqdm.write(f'  {norm_mode[:12]}  gs={gs}  sample={idx}  '
                   f'int={r[0]:.1f}  vel={r[1]:.2f}  wid={r[2]:.2f}')
        results[norm_mode].append(r)

n_gs = len(GRAD_SCALES)
for nm in results:
    arr = np.stack(results[nm])
    results[nm] = arr.reshape(n_gs, N_TEST, arr.shape[-1])

np.save(f'{OUT_DIR}/grad_scale_sweep.npy', results)

# ── summary table ─────────────────────────────────────────────────────────────
print(f'\n{"":22} ' + '  '.join(f'gs={gs:<5}' for gs in GRAD_SCALES))
for norm_mode, arr in results.items():
    print(f'\n{norm_mode}')
    for c, lbl in enumerate(CH_LABELS):
        row = '  '.join(f'{arr[i, :, c].mean():7.2f}' for i in range(n_gs))
        print(f'  {lbl:<16} {row}')

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
colors  = {'global_logz': 'steelblue', 'persample_linear': 'tomato'}
markers = {'global_logz': 'o', 'persample_linear': 's'}

for norm_mode, arr in results.items():
    for c, (ax, lbl) in enumerate(zip(axes, CH_LABELS)):
        ax.errorbar(GRAD_SCALES, arr[:, :, c].mean(axis=1), yerr=arr[:, :, c].std(axis=1),
                    label=norm_mode, color=colors[norm_mode],
                    marker=markers[norm_mode], capsize=4)

for ax, lbl in zip(axes, CH_LABELS):
    ax.set_xlabel('grad_scale')
    ax.set_ylabel(f'RMSE {lbl}')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_title(lbl)

fig.suptitle(f'grad_scale sweep — milestone={MILESTONE}  n_test={N_TEST}', fontsize=11)
fig.tight_layout()
out = f'{OUT_DIR}/grad_scale_sweep.png'
fig.savefig(out, dpi=150)
plt.close(fig)
print(f'\nsaved {out}')