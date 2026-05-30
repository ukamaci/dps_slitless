import os
import torch
import glob
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import matplotlib.pyplot as plt
import numpy as np
from statistics import NormalDist

from denoising_diffusion_pytorch.normalization import make_normalization

# ── config ────────────────────────────────────────────────────────────────────
modellist   = [1, 5, 10, 15, 20, 25, 30, 40, 50]
num_samples = 100

runs = [
    ('global_logz',      'training_results/exp_norm_logz_dset6_lr5e-6'),
    ('persample_linear', 'training_results/exp_norm_persample_dset6_lr5e-6'),
]

DATA_DIR = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data/train'
OUT_DIR  = 'experiments/generator_quality/outputs'
# ─────────────────────────────────────────────────────────────────────────────

SPEEDOFLIGHT = 299792.458
WAVELENGTH   = 195.117937907451
W_FAC        = SPEEDOFLIGHT / WAVELENGTH   # Å → km/s

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gaussmatch_score(means1, stds1, means2, stds2):
    scores = np.zeros(len(means1))
    for i in range(len(means1)):
        scores[i] = NormalDist(mu=means1[i], sigma=stds1[i]).overlap(
                    NormalDist(mu=means2[i], sigma=stds2[i]))
    return scores


def persample_norm_int(imgs):
    """Per-sample normalize intensity to [0,1]: imgs (N,H,W) or (N,C,H,W) ch0."""
    if imgs.ndim == 4:
        imgs = imgs[:, 0]
    mx = imgs.reshape(len(imgs), -1).max(axis=1, keepdims=True)[:, :, None]
    return (imgs / np.maximum(mx, 1.0)).ravel()


# ── EIS reference stats from dset_v6 train (physical units) ──────────────────
files = sorted(glob.glob(DATA_DIR + '/data*.npy'))
eis_int, eis_vel, eis_wid = [], [], []
eis_int_norm = []   # per-sample normalized intensity for shape comparison
for f in files:
    d = np.load(f, allow_pickle=True).item()
    eis_int.append(d['int'].ravel())
    eis_vel.append(d['vel'].ravel())
    eis_wid.append(d['width'].ravel() * W_FAC)   # Å → km/s
    eis_int_norm.append(persample_norm_int(d['int'][None]))

eis_int      = np.concatenate(eis_int)
eis_vel      = np.concatenate(eis_vel)
eis_wid      = np.concatenate(eis_wid)
eis_int_norm = np.concatenate(eis_int_norm)

# vel/width compared in physical units; int only meaningful for global_logz
means_eis = [eis_int.mean(), eis_vel.mean(), eis_wid.mean()]
stds_eis  = [eis_int.std(),  eis_vel.std(),  eis_wid.std()]

print(f'EIS reference: int={means_eis[0]:.1f}±{stds_eis[0]:.1f} erg/cm²/s/sr  '
      f'vel={means_eis[1]:.2f}±{stds_eis[1]:.2f} km/s  '
      f'width={means_eis[2]:.2f}±{stds_eis[2]:.2f} km/s')

# ── model ─────────────────────────────────────────────────────────────────────
model = Unet(
    channels  = 3,
    dim       = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
).to(device)

os.makedirs(OUT_DIR, exist_ok=True)

# ── sweep ─────────────────────────────────────────────────────────────────────
all_scores   = {}   # norm_mode → (n_milestones, 3)
all_int_norm = {}   # norm_mode → {modelnum: pixels}

for norm_mode, run_folder in runs:
    normalization = make_normalization(norm_mode, rec_mode='all')

    diffusion = GaussianDiffusion(
        model,
        mode       = 'all',
        image_size = 64,
        timesteps  = 1000,
        sampling_timesteps = 250,
        beta_schedule = 'cosine',
        clip_denoised = (-5., 5.),
        normalization = normalization,
        device = device,
    )

    scores = []
    int_norm_by_ms = {}   # modelnum → per-sample-normalised intensity pixels
    for modelnum in modellist:
        ckpt_path = f'{run_folder}/model-{modelnum}.pt'
        if not os.path.exists(ckpt_path):
            print(f'  [skip] {norm_mode} ms={modelnum} not found'); scores.append([np.nan]*3); continue

        data = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = {k[6:]: v for k, v in data['model'].items() if k.startswith('model.')}
        model.load_state_dict(state)
        model.eval()

        with torch.inference_mode():
            s = diffusion.sample(batch_size=num_samples).cpu().numpy()  # (N,3,H,W) physical

        s[:, 2] *= W_FAC   # width Å → km/s
        int_norm_by_ms[modelnum] = persample_norm_int(s)

        # int: use per-sample normalised — raw DN meaningless for persample_linear
        means_ddpm = [int_norm_by_ms[modelnum].mean(), s[:, 1].mean(), s[:, 2].mean()]
        stds_ddpm  = [int_norm_by_ms[modelnum].std(),  s[:, 1].std(),  s[:, 2].std()]
        means_ref  = [eis_int_norm.mean(), means_eis[1], means_eis[2]]
        stds_ref   = [eis_int_norm.std(),  stds_eis[1],  stds_eis[2]]

        sc = gaussmatch_score(means_ref, stds_ref, means_ddpm, stds_ddpm)
        scores.append(sc)
        print(f'  {norm_mode}  ms={modelnum:>3}  '
              f'int={sc[0]:.3f}  vel={sc[1]:.3f}  width={sc[2]:.3f}')

    all_scores[norm_mode] = np.array(scores)   # (n_ms, 3)
    all_int_norm[norm_mode] = int_norm_by_ms

# ── save results ──────────────────────────────────────────────────────────────
np.save(f'{OUT_DIR}/evaluate_results.npy', {
    'config': {'modellist': modellist, 'num_samples': num_samples, 'runs': runs},
    'gaussmatch': all_scores,
    'int_norm': all_int_norm,
    'eis_stats': {'means': means_eis, 'stds': stds_eis},
})

# ── plot: gaussmatch vs milestone ─────────────────────────────────────────────
steps  = [ms * 1000 for ms in modellist]
colors = {'global_logz': 'steelblue', 'persample_linear': 'tomato'}
ch_labels = ['int (norm) *', 'vel (km/s)', 'width (km/s)']

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for norm_mode, scores in all_scores.items():
    for c, ax in enumerate(axes):
        ax.plot(steps, scores[:, c], '-o', label=norm_mode, color=colors[norm_mode])

for ax, lbl in zip(axes, ch_labels):
    ax.set_xlabel('Training step')
    ax.set_ylabel('Gaussmatch score')
    ax.set_title(lbl)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

fig.suptitle('Gaussmatch vs checkpoint  (* int not comparable for persample_linear)', fontsize=10)
fig.tight_layout()
plt.savefig(f'{OUT_DIR}/gaussmatch_vs_milestone.png', dpi=150)
plt.show()

# ── plot: per-sample normalised intensity histograms ─────────────────────────
plot_ms = [1, 10, 50]   # subset of milestones to show
bins = np.linspace(0, 1, 80)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(eis_int_norm, bins=bins, density=True, alpha=0.5, color='black', label='EIS train')

for norm_mode, int_norm_by_ms in all_int_norm.items():
    for ms in plot_ms:
        if ms not in int_norm_by_ms:
            continue
        ax.hist(int_norm_by_ms[ms], bins=bins, density=True, alpha=0.35,
                color=colors[norm_mode],
                label=f'{norm_mode} ms={ms}')

ax.set_xlabel('per-sample normalised intensity  [x / max(x)]')
ax.set_ylabel('density')
ax.set_title('Intensity shape comparison (absolute scale removed)')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig(f'{OUT_DIR}/int_shape_histogram.png', dpi=150)
plt.show()