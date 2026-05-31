"""
Normalization experiment analysis.

Edit the config block below, then: python experiments/normalization/analyze.py

Produces:
  outputs/loss_curves.png          — training loss comparison
  outputs/samples_logz.png         — periodic samples at milestones 10/25/50
  outputs/samples_persample.png
  outputs/recon_logz_idx{i}.png    — DPS reconstruction grids per test sample
  outputs/recon_persample_idx{i}.png
  outputs/rmse_summary.npy         — dict with all RMSEs
  outputs/rmse_table.txt           — printable summary table
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization
from slitless.forward import forward_op_torch

# ── constants ─────────────────────────────────────────────────────────────────
SPEEDOFLIGHT     = 299792.458
WAVELENGTH       = 195.117937907451
W_FAC            = SPEEDOFLIGHT / WAVELENGTH   # Å → km/s  (~1536.5)
DISPERSION_SCALE = 0.022275          # Å/pixel  (EIS: 13.5 µm pixel / (1/1.65 µm/mÅ) / 1000)
VEL_TO_PIX       = WAVELENGTH / SPEEDOFLIGHT / DISPERSION_SCALE  # km/s → pixels (~0.02922)
WIDTH_TO_PIX      = 1.0 / DISPERSION_SCALE                        # Å    → pixels (~44.89)

DATA_DIR   = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data/test'
RUN_LOGZ   = os.path.join(REPO_ROOT, 'training_results/exp_norm_logz_dset6_lr5e-6')
RUN_PS     = os.path.join(REPO_ROOT, 'training_results/exp_norm_persample_dset6_lr5e-6')
OUT_DIR    = os.path.join(REPO_ROOT, 'experiments/normalization/outputs')

# ── config ────────────────────────────────────────────────────────────────────
MILESTONE   = 10
N_TEST      = 5
DPS_SAMPLES = 5
RUN_DPS     = True    # set False to only plot loss curves + periodic samples
# ─────────────────────────────────────────────────────────────────────────────

CMAPS = ['hot', 'seismic', 'plasma']
CH_LABELS = ['int (erg/cm²/s/sr)', 'vel (km/s)', 'width (km/s)']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── helpers ───────────────────────────────────────────────────────────────────
def rmse(a, b, axis=(-1, -2)):
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))


def to_physical(samples):
    """Convert width channel from Å to km/s in-place (samples: N,C,H,W numpy)."""
    out = samples.copy()
    out[:, 2] *= W_FAC
    return out


def load_test_sample(idx):
    files = sorted(glob.glob(DATA_DIR + '/data*.npy'))
    d = np.load(files[idx], allow_pickle=True).item()
    orders = [0, -1, 1]
    meas_np  = np.stack([d[f'meas_{o}'] for o in orders])[None].astype(np.float32)
    true_np  = np.stack([d['int'], d['vel'], d['width']])[None].astype(np.float32)
    return meas_np, true_np


def build_model(run_folder, milestone, norm_mode, meas, true):
    normalization = make_normalization(norm_mode, rec_mode='all')
    if norm_mode == 'persample_linear':
        normalization.set_infer_scale(torch.tensor(meas[:, 0]).max())

    model = Unet(channels=3, dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(device)
    ckpt = torch.load(f'{run_folder}/model-{milestone}.pt', map_location=device, weights_only=True)
    state = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
    model.load_state_dict(state)
    model.eval()

    meas_t = torch.tensor(meas).to(device)
    true_t = torch.tensor(true).to(device)

    def forward_op(x, device=None):
        return forward_op_torch(true_intensity=x[:, 0],
                                true_doppler=x[:, 1] * VEL_TO_PIX,
                                true_linewidth=x[:, 2] * WIDTH_TO_PIX, device=device)

    diffusion = GaussianDiffusion(
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
        grad_scale=torch.tensor([1.0]).to(device),
        forward_op=forward_op,
        device=device,
        normalization=normalization,
    )
    return diffusion


def run_dps(run_folder, milestone, norm_mode, sample_idx, dps_samples):
    meas_np, true_np = load_test_sample(sample_idx)
    diffusion = build_model(run_folder, milestone, norm_mode, meas_np, true_np)

    samples, *_ = diffusion.sample(batch_size=dps_samples)

    samples = samples.detach().cpu().numpy()   # (S, C, H, W) physical
    samples = to_physical(samples)
    true_ph = to_physical(true_np)             # (1, C, H, W)

    mean_r  = samples.mean(axis=0, keepdims=True)  # (1, C, H, W)
    rmse_mean  = rmse(true_ph[0], mean_r[0]).squeeze()     # (C,)
    rmse_per   = np.stack([rmse(true_ph[0], s).squeeze() for s in samples])  # (S, C)

    return samples, true_ph[0], meas_np[0], rmse_mean, rmse_per


# ── figure: loss curves ───────────────────────────────────────────────────────
def plot_loss_curves():
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, run in [('global_logz', RUN_LOGZ), ('persample_linear', RUN_PS)]:
        path = f'{run}/train_loss.npy'
        if not os.path.exists(path):
            print(f'  [warn] missing {path}'); continue
        loss = np.load(path)
        steps = np.arange(1, len(loss) + 1) * 1   # one entry per step
        ax.semilogy(steps, loss, label=label, alpha=0.8)
        # smoothed
        k = 200
        smooth = np.convolve(loss, np.ones(k)/k, mode='valid')
        ax.semilogy(np.arange(k, len(loss) + 1), smooth, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training loss — normalization comparison')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    out = f'{OUT_DIR}/loss_curves.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


# ── figure: periodic samples at milestone grid ────────────────────────────────
def plot_periodic_samples(run_folder, norm_mode, milestones=(10, 25, 50)):
    milestones = [m for m in milestones if
                  os.path.exists(f'{run_folder}/periodic_samples/sample-{m}.png')]
    if not milestones:
        print(f'  [warn] no periodic sample PNGs found in {run_folder}'); return

    fig, axes = plt.subplots(1, len(milestones), figsize=(5 * len(milestones), 5))
    if len(milestones) == 1:
        axes = [axes]
    for ax, m in zip(axes, milestones):
        img = plt.imread(f'{run_folder}/periodic_samples/sample-{m}.png')
        ax.imshow(img)
        ax.set_title(f'step {m*1000:,}')
        ax.axis('off')
    fig.suptitle(f'Periodic samples — {norm_mode}', fontsize=13)
    fig.tight_layout()
    out = f'{OUT_DIR}/samples_{norm_mode}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


# ── figure: DPS reconstruction grid ──────────────────────────────────────────
def plot_recon(samples, true, meas, norm_mode, sample_idx, rmse_mean):
    """samples: (S,C,H,W), true: (C,H,W), meas: (K,H,W) — all physical, width in km/s."""
    n_show = min(3, len(samples))
    cols   = ['Meas (ch0)', 'True'] + [f'Sample {i+1}' for i in range(n_show)] + ['Post. mean']
    n_cols = len(cols)

    fig, ax = plt.subplots(3, n_cols, figsize=(2.5 * n_cols, 8))
    mean_r  = samples.mean(axis=0)

    recs    = [meas, true] + list(samples[:n_show]) + [mean_r]
    vmins   = [true[c].min() for c in range(3)]
    vmaxs   = [true[c].max() for c in range(3)]

    for col, (rec, title) in enumerate(zip(recs, cols)):
        for row in range(3):
            data = rec[row] if rec.ndim == 3 and rec.shape[0] > row else rec[0]
            use_clim = col > 0
            im = ax[row, col].imshow(data, cmap=CMAPS[row],
                                     vmin=vmins[row] if use_clim else None,
                                     vmax=vmaxs[row] if use_clim else None)
            ax[row, col].set_xticks([]); ax[row, col].set_yticks([])
            if row == 0:
                ax[row, col].set_title(title, fontsize=9)
            if col == 0:
                ax[row, col].set_ylabel(CH_LABELS[row], fontsize=8)
            fig.colorbar(im, ax=ax[row, col], orientation='horizontal',
                         location='top', pad=0.02, fraction=0.046)

    rmse_str = f'int={rmse_mean[0]:.1f} erg/cm²/s/sr  vel={rmse_mean[1]:.2f} km/s  width={rmse_mean[2]:.2f} km/s'
    fig.suptitle(f'{norm_mode} | sample_idx={sample_idx} | RMSE (post. mean): {rmse_str}', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f'{OUT_DIR}/recon_{norm_mode}_idx{sample_idx}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved {out}')


# ── main ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

print('Plotting loss curves...')
plot_loss_curves()

print('Plotting periodic samples...')
for run, nm in [(RUN_LOGZ, 'global_logz'), (RUN_PS, 'persample_linear')]:
    plot_periodic_samples(run, nm, milestones=(10, 25, 50))

if RUN_DPS:
    all_rmse = {}

    for run_folder, norm_mode in [(RUN_LOGZ, 'global_logz'), (RUN_PS, 'persample_linear')]:
        print(f'\nRunning DPS — {norm_mode} — milestone {MILESTONE}')
        rmse_list = []
        for idx in range(N_TEST):
            print(f'  test sample {idx}...')
            samples, true, meas, rmse_mean, rmse_per = run_dps(
                run_folder, MILESTONE, norm_mode, idx, DPS_SAMPLES)
            rmse_list.append(rmse_mean)
            plot_recon(samples, true, meas, norm_mode, idx, rmse_mean)

        all_rmse[norm_mode] = np.stack(rmse_list)   # (n_test, 3)

    np.save(f'{OUT_DIR}/rmse_summary.npy', all_rmse)

    lines = []
    lines.append(f'\n{"Method":<22} {"int RMSE (erg/cm²/s/sr)":>15} {"vel RMSE (km/s)":>17} {"width RMSE (km/s)":>19}')
    lines.append('-' * 76)
    for nm, arr in all_rmse.items():
        m = arr.mean(axis=0)
        s = arr.std(axis=0)
        lines.append(f'{nm:<22} {m[0]:>8.1f}±{s[0]:<5.1f} {m[1]:>10.2f}±{s[1]:<5.2f} {m[2]:>12.2f}±{s[2]:<5.2f}')

    table = '\n'.join(lines)
    print(table)
    with open(f'{OUT_DIR}/rmse_table.txt', 'w') as f:
        f.write(f'Milestone: {MILESTONE}  n_test: {N_TEST}  dps_samples: {DPS_SAMPLES}\n')
        f.write(table + '\n')
    print(f'\n  saved {OUT_DIR}/rmse_table.txt')
else:
    print('Skipping DPS (RUN_DPS=False).')
