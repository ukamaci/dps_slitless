"""
Memorization → generalization experiment (Kadkhodaie et al., ICLR 2024, Fig. 2).

Two unconditional DDPMs are trained on disjoint partitions S1 / S2 of the dset_v6
training set, at three partition sizes (1/2, 1/4, 1/8 of the full set). If the
models memorize, each reproduces images from its own training partition and the
two models disagree with each other. If they generalize, the two models converge
to the same density: they produce near-identical samples from shared init noise,
and those samples no longer match any single training image.

This script samples from each model (shared init noise across S1/S2 within a
column so the pair is comparable), finds the closest training image to each
sample in that model's own partition, and reproduces Figure 2:

  Top   — 4xNcol image grid: closest S1 train img / S1 sample / S2 sample /
          closest S2 train img, one column per partition size.
  Bottom— per-column cosine-similarity histograms: blue = between the two
          models' paired samples (model variance), orange = between a generated
          sample and the closest image in its training partition (memorization).

Cosine similarity and "closest image" are computed in the model's *normalized*
space (global_logz) so the three physical channels (int/vel/width) are weighted
comparably. The image panels display one channel at a time (config CHANNEL);
a figure is written per channel.

Run config (norm_mode, partno/partnum, sampling_timesteps, dataset_path) is read
from each run's config.json — not duplicated here.

Outputs (experiments/generalization_memorization/outputs/):
  generalization_memorization_int.png    — Figure-2 analog, intensity channel
  generalization_memorization_vel.png    —   "   velocity channel
  generalization_memorization_width.png  —   "   line-width channel
  results.npy                            — dict: per-column cosine arrays + config
  summary.txt                            — mean pairwise / closest cosine per column

Edit the config block, then: python experiments/generalization_memorization/analyze.py
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import partition_files

# ── runs: the disjoint-partition models (hardcoded — experiment record) ─────────
# Partition sizes 1/2..1/8 (≈11400/5700/2850 patches) and 1/32..1/2048
# (≈710/176/45/15 patches), each with two disjoint partitions S1 (partno 1) and
# S2 (partno 2). The 1/2..1/8 runs save model-1..10.pt (milestone 10 = final); the
# 1/32..1/2048 runs save a single model-final.pt — find_checkpoint() handles both.
TR = os.path.join(REPO_ROOT, 'training_results')
RUNS = [
    f'{TR}/2026_06_01__06_12_41_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_None_dsize_1v2',
    f'{TR}/2026_06_01__10_38_13_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_None_dsize_2v2',
    # 1/4 partition omitted so column sizes follow a clean ×4 sweep (2,8,32,128,512,2048):
    # f'{TR}/2026_06_01__15_03_52_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_None_dsize_1v4',
    # f'{TR}/2026_06_01__17_18_07_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_None_dsize_2v4',
    f'{TR}/2026_06_01__19_32_14_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_None_dsize_1v8',
    f'{TR}/2026_06_01__20_40_53_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_None_dsize_2v8',
    f'{TR}/2026_06_03__00_45_20_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_1v32',
    f'{TR}/2026_06_03__01_05_12_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_2v32',
    f'{TR}/2026_06_03__01_25_09_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_1v128',
    f'{TR}/2026_06_03__01_31_47_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_2v128',
    f'{TR}/2026_06_03__01_38_16_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_1v512',
    f'{TR}/2026_06_03__01_41_44_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_2v512',
    f'{TR}/2026_06_03__01_45_14_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_1v2048',
    f'{TR}/2026_06_03__01_49_23_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_2v2048',
]
OUT_DIR = os.path.join(REPO_ROOT, 'experiments/generalization_memorization/outputs')

# ── config ──────────────────────────────────────────────────────────────────
NUM_PAIRS          = 30       # generated samples per model -> histogram counts
SAMPLING_TIMESTEPS = 250      # DDIM, eta=0 (deterministic): shared init noise -> comparable pair
SEED               = 0        # base seed; column c uses SEED+c so S1/S2 share init noise
USE_EMA            = True     # sample the EMA weights (as training-time periodic samples do)
CLOSEST_SUBSAMPLE  = None     # cap train images searched for the closest match (None = all)
HIST_BINS          = 40
# ────────────────────────────────────────────────────────────────────────────

CMAPS     = {'int': 'hot', 'vel': 'seismic', 'width': 'plasma'}
CH_IDX    = {'int': 0, 'vel': 1, 'width': 2}
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DSHAPE    = (3, 64, 64)


# ── model / sampling ──────────────────────────────────────────────────────────
def read_config(run_folder):
    with open(os.path.join(run_folder, 'config.json')) as f:
        return json.load(f)


def find_checkpoint(run_folder):
    """Resolve a run's final checkpoint: prefer model-final.pt (save_every=None runs),
    else the highest-numbered milestone model-{n}.pt."""
    final = os.path.join(run_folder, 'model-final.pt')
    if os.path.exists(final):
        return final
    nums = [int(os.path.basename(p)[6:-3])
            for p in glob.glob(os.path.join(run_folder, 'model-*.pt'))
            if os.path.basename(p)[6:-3].isdigit()]
    if not nums:
        raise FileNotFoundError(f'no model-*.pt checkpoint in {run_folder}')
    return os.path.join(run_folder, f'model-{max(nums)}.pt')


def load_unet(ckpt_path, use_ema):
    """Load the (EMA) U-Net weights from a checkpoint into a fresh Unet."""
    model = Unet(channels=3, dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if use_ema:
        # EMA wraps the GaussianDiffusion: U-Net params live under 'ema_model.model.'
        pref = 'ema_model.model.'
        state = {k[len(pref):]: v for k, v in ckpt['ema'].items() if k.startswith(pref)}
    else:
        state = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
    model.load_state_dict(state)
    model.eval()
    return model


def build_diffusion(model, cfg):
    return GaussianDiffusion(
        model,
        mode='all',
        image_size=cfg['image_size'],
        timesteps=cfg['timesteps'],
        sampling_timesteps=SAMPLING_TIMESTEPS,
        beta_schedule=cfg['beta_schedule'],
        clip_denoised=tuple(cfg['clip_denoised']),
        normalization=make_normalization(cfg['norm_mode'], rec_mode='all'),
        device=device,
    )


def sample_model(diffusion, n, seed):
    """n physical samples (n,3,64,64); seeded so a paired model with the same seed
    starts from the same init noise (DDIM eta=0 is deterministic given x_T)."""
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        s = diffusion.sample(batch_size=n)
    return s.detach().cpu().numpy()


# ── data ──────────────────────────────────────────────────────────────────────
def load_partition_images(dataset_path, partno, partnum, subsample=None):
    """Physical (N,3,64,64) for one leakage-free partition (matches what trained it)."""
    files = partition_files(glob.glob(dataset_path + '/data*.npy'), partno, partnum)
    if subsample is not None and subsample < len(files):
        files = list(np.array(files)[np.linspace(0, len(files) - 1, subsample).astype(int)])
    imgs = np.empty((len(files), *DSHAPE), dtype=np.float32)
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True).item()
        imgs[i] = np.stack([d['int'], d['vel'], d['width']])
    return imgs


def norm_flat(norm, x_np):
    """Physical (M,3,64,64) -> normalized, flattened, L2-normalized rows (M, 3*64*64)."""
    with torch.no_grad():
        v = norm.forward(torch.tensor(x_np, dtype=torch.float32)).reshape(len(x_np), -1).numpy()
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v


def closest(sample_vecs, train_vecs):
    """For each sample row, max cosine similarity over train rows + its argmax index."""
    sims = sample_vecs @ train_vecs.T          # (K, N), rows are unit-norm -> cosine
    return sims.max(axis=1), sims.argmax(axis=1)


# ── main ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
norm = make_normalization('global_logz', rec_mode='all')   # shared frame for all cosine sims

# group runs into columns by partnum (descending partnum == ascending N == left->right)
by_partnum = {}
for folder in RUNS:
    cfg = read_config(folder)
    by_partnum.setdefault(cfg['partnum'], {})[cfg['partno']] = folder
partnums = sorted(by_partnum, reverse=True)   # [8, 4, 2]
ncol = len(partnums)

columns = []   # per-column dict of everything needed for plotting + results
for c, pn in enumerate(partnums):
    folder1, folder2 = by_partnum[pn][1], by_partnum[pn][2]
    cfg1, cfg2 = read_config(folder1), read_config(folder2)
    N = 0  # training-partition size (filled below)

    print(f'\n=== partition 1/{pn} (column {c + 1}/{ncol}) ===')
    print('  sampling S1, S2 (shared init noise)...')
    diff1 = build_diffusion(load_unet(find_checkpoint(folder1), USE_EMA), cfg1)
    s1 = sample_model(diff1, NUM_PAIRS, SEED + c)
    del diff1
    diff2 = build_diffusion(load_unet(find_checkpoint(folder2), USE_EMA), cfg2)
    s2 = sample_model(diff2, NUM_PAIRS, SEED + c)
    del diff2
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    s1v, s2v = norm_flat(norm, s1), norm_flat(norm, s2)
    pair_cos = (s1v * s2v).sum(axis=1)                       # model-variance (blue)

    print('  loading training partitions + closest-image search...')
    tr1 = load_partition_images(cfg1['dataset_path'], 1, pn, CLOSEST_SUBSAMPLE)
    tr2 = load_partition_images(cfg2['dataset_path'], 2, pn, CLOSEST_SUBSAMPLE)
    N = (len(tr1) + len(tr2)) // 2
    cclose1, arg1 = closest(s1v, norm_flat(norm, tr1))       # memorization (orange)
    cclose2, arg2 = closest(s2v, norm_flat(norm, tr2))

    columns.append(dict(
        partnum=pn, N=N,
        s1=s1, s2=s2,
        closest_s1=tr1[arg1[0]], closest_s2=tr2[arg2[0]],   # for representative pair (idx 0)
        pair_cos=pair_cos,
        closest_cos_s1=cclose1, closest_cos_s2=cclose2,
    ))
    print(f'  N≈{N}  mean pair cos={pair_cos.mean():.3f}  '
          f'mean closest cos={0.5 * (cclose1.mean() + cclose2.mean()):.3f}')
    del tr1, tr2


# ── figure (one per channel) ──────────────────────────────────────────────────
ROW_LABELS = ['Closest image from $S_1$', 'Generated by models trained on $S_1$',
              'Generated by models trained on $S_2$', 'Closest image from $S_2$']


def make_figure(channel):
    ci, cmap = CH_IDX[channel], CMAPS[channel]
    fig = plt.figure(figsize=(2.4 * ncol + 1.4, 12))
    gs = fig.add_gridspec(5, ncol, height_ratios=[1, 1, 1, 1, 1.25], hspace=0.12, wspace=0.08)

    for c, col in enumerate(columns):
        rep = [col['closest_s1'], col['s1'][0], col['s2'][0], col['closest_s2']]
        for r in range(4):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(rep[r][ci], cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'$N$={col["N"]}  (1/{col["partnum"]})', fontsize=11)
            if c == 0:
                ax.set_ylabel(ROW_LABELS[r], fontsize=8)

        # bottom: cosine-similarity histograms for this column
        ax = fig.add_subplot(gs[4, c])
        orange = np.concatenate([col['closest_cos_s1'], col['closest_cos_s2']])
        ax.hist(col['pair_cos'], bins=HIST_BINS, range=(0, 1), color='tab:blue',
                alpha=0.7, label='samples from two denoisers')
        ax.hist(orange, bins=HIST_BINS, range=(0, 1), color='tab:orange',
                alpha=0.7, label='sample and closest train image')
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xlabel('cosine similarity', fontsize=8)
        if c == 0:
            ax.legend(fontsize=6, loc='upper left')

    fig.suptitle(f'Memorization → generalization  ({channel} channel shown; '
                 f'cosine over all 3 channels in global_logz space)', fontsize=11)
    out = f'{OUT_DIR}/generalization_memorization_{channel}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


print('\nBuilding figures...')
for ch in ('int', 'vel', 'width'):
    make_figure(ch)


# ── save raw results + summary ────────────────────────────────────────────────
results = dict(
    config=dict(NUM_PAIRS=NUM_PAIRS,
                SAMPLING_TIMESTEPS=SAMPLING_TIMESTEPS, SEED=SEED, USE_EMA=USE_EMA,
                CLOSEST_SUBSAMPLE=CLOSEST_SUBSAMPLE, runs=RUNS),
    columns=[{k: v for k, v in col.items() if k not in ('s1', 's2')} for col in columns],
)
np.save(f'{OUT_DIR}/results.npy', results)

lines = [f'Memorization→generalization | final checkpoints | {NUM_PAIRS} pairs/col '
         f'| {SAMPLING_TIMESTEPS} DDIM steps | EMA={USE_EMA}', '',
         f'{"partition":>10} {"N":>7} {"mean pair cos":>15} {"mean closest cos":>18}',
         '-' * 54]
for col in columns:
    cc = 0.5 * (col['closest_cos_s1'].mean() + col['closest_cos_s2'].mean())
    lines.append(f'{"1/" + str(col["partnum"]):>10} {col["N"]:>7} '
                 f'{col["pair_cos"].mean():>15.3f} {cc:>18.3f}')
summary = '\n'.join(lines)
print('\n' + summary)
with open(f'{OUT_DIR}/summary.txt', 'w') as f:
    f.write(summary + '\n')
print(f'\n  saved {OUT_DIR}/results.npy and {OUT_DIR}/summary.txt')
