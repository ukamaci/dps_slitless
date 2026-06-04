"""
memorization2048 analysis.

Question: trained to convergence (25k steps, matched compute), do the two disjoint
single-scan N=15 models memorize? Memorization signature = (a) generated samples
become sharp and match the model's own 15 training patches (closest-train cosine
high, AND visually identical), and (b) the two models disagree with each other
(pair cosine low) because they memorized different scans. Generalization would
instead show high pair cosine and samples unlike any training patch.

Because we checkpoint every 2.5k steps, we can watch *when* this emerges. Cosine /
closest are computed in the model's normalized (global_logz) space over all 3
channels; image panels use shared per-channel color scales so visual comparison
is fair (unlike the auto-scaled main-sweep figure).

Outputs (experiments/memorization2048/outputs/):
  trainset_int.png                  — the 15 training patches of S1 and S2 (intensity)
  memorization_grid_{int,vel,width}.png  — closest-S1 / S1 sample / S2 sample / closest-S2
                                           at the final checkpoint, shared color scale
  convergence.png                   — mean pair & closest cosine vs training steps
  results.npy                       — per-milestone cosine arrays + config
  summary.txt

Run (after training): python experiments/memorization2048/analyze.py
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

# ── runs: resolve the two _mem25k disjoint-partition runs (experiment record) ───
TR = os.path.join(REPO_ROOT, 'training_results')


def _resolve(tag):
    hits = sorted(glob.glob(f'{TR}/*dsize_{tag}_mem25k'))
    assert len(hits) >= 1, f'no run folder matching *dsize_{tag}_mem25k in {TR}'
    return hits[-1]   # latest if re-run


RUN_S1 = _resolve('1v2048')
RUN_S2 = _resolve('2v2048')
OUT_DIR = os.path.join(REPO_ROOT, 'experiments/memorization2048/outputs')

# ── config ──────────────────────────────────────────────────────────────────
NUM_PAIRS          = 16       # generated samples per model per milestone
SAMPLING_TIMESTEPS = 250      # DDIM, eta=0 (deterministic): shared init noise -> comparable pair
SEED               = 0        # both models share init noise (same seed) at each milestone
USE_EMA            = True
N_SHOW             = 6        # columns in the memorization grid
HIST_BINS          = 30
# ────────────────────────────────────────────────────────────────────────────

CMAPS  = {'int': 'hot', 'vel': 'seismic', 'width': 'plasma'}
CH_IDX = {'int': 0, 'vel': 1, 'width': 2}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DSHAPE = (3, 64, 64)


# ── helpers ───────────────────────────────────────────────────────────────────
def read_config(run_folder):
    with open(os.path.join(run_folder, 'config.json')) as f:
        return json.load(f)


def milestones_of(run_folder):
    nums = [int(os.path.basename(p)[6:-3])
            for p in glob.glob(os.path.join(run_folder, 'model-*.pt'))
            if os.path.basename(p)[6:-3].isdigit()]
    return sorted(nums)


def load_unet(run_folder, milestone, use_ema):
    model = Unet(channels=3, dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(device)
    ckpt = torch.load(f'{run_folder}/model-{milestone}.pt', map_location=device, weights_only=True)
    pref = 'ema_model.model.' if use_ema else None
    if use_ema:
        state = {k[len(pref):]: v for k, v in ckpt['ema'].items() if k.startswith(pref)}
    else:
        state = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
    model.load_state_dict(state)
    model.eval()
    return model


def build_diffusion(model, cfg):
    return GaussianDiffusion(
        model, mode='all', image_size=cfg['image_size'], timesteps=cfg['timesteps'],
        sampling_timesteps=SAMPLING_TIMESTEPS, beta_schedule=cfg['beta_schedule'],
        clip_denoised=tuple(cfg['clip_denoised']),
        normalization=make_normalization(cfg['norm_mode'], rec_mode='all'), device=device)


def sample_model(diffusion, n, seed):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        return diffusion.sample(batch_size=n).detach().cpu().numpy()


def load_partition_images(dataset_path, partno, partnum):
    files = partition_files(glob.glob(dataset_path + '/data*.npy'), partno, partnum)
    imgs = np.empty((len(files), *DSHAPE), dtype=np.float32)
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True).item()
        imgs[i] = np.stack([d['int'], d['vel'], d['width']])
    return imgs


def norm_flat(norm, x_np):
    with torch.no_grad():
        v = norm.forward(torch.tensor(x_np, dtype=torch.float32)).reshape(len(x_np), -1).numpy()
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v


def closest(sample_vecs, train_vecs):
    sims = sample_vecs @ train_vecs.T
    return sims.max(axis=1), sims.argmax(axis=1)


# ── main ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
norm = make_normalization('global_logz', rec_mode='all')
cfg1, cfg2 = read_config(RUN_S1), read_config(RUN_S2)

# the two single-scan training sets (15 patches each)
tr1 = load_partition_images(cfg1['dataset_path'], cfg1['partno'], cfg1['partnum'])
tr2 = load_partition_images(cfg2['dataset_path'], cfg2['partno'], cfg2['partnum'])
tr1v, tr2v = norm_flat(norm, tr1), norm_flat(norm, tr2)
print(f'S1 train patches: {len(tr1)}   S2 train patches: {len(tr2)}')

mics = sorted(set(milestones_of(RUN_S1)) & set(milestones_of(RUN_S2)))   # common checkpoints
assert mics, 'no common checkpoints between S1 and S2'
steps = [m * cfg1['save_every'] for m in mics]   # save_every is in (epoch==step) units here

traj = []   # per-milestone summary
final = None
for m, st in zip(mics, steps):
    diff1 = build_diffusion(load_unet(RUN_S1, m, USE_EMA), cfg1)
    s1 = sample_model(diff1, NUM_PAIRS, SEED)
    del diff1
    diff2 = build_diffusion(load_unet(RUN_S2, m, USE_EMA), cfg2)
    s2 = sample_model(diff2, NUM_PAIRS, SEED)
    del diff2
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    s1v, s2v = norm_flat(norm, s1), norm_flat(norm, s2)
    pair = (s1v * s2v).sum(axis=1)
    c1, a1 = closest(s1v, tr1v)
    c2, a2 = closest(s2v, tr2v)
    traj.append(dict(milestone=m, step=st, pair_cos=pair,
                     closest_cos_s1=c1, closest_cos_s2=c2))
    print(f'  step {st:6d}: pair={pair.mean():+.3f}  closest={0.5*(c1.mean()+c2.mean()):.3f}')
    if m == mics[-1]:
        final = dict(s1=s1, s2=s2, near1=tr1[a1], near2=tr2[a2])


# ── figure: training-set overview (intensity) ─────────────────────────────────
def fig_trainset():
    n = max(len(tr1), len(tr2))
    fig, ax = plt.subplots(2, n, figsize=(1.3 * n, 3.0))
    vmin, vmax = tr1[:, 0].min(), tr1[:, 0].max()
    for row, tr in enumerate((tr1, tr2)):
        for j in range(n):
            a = ax[row, j]
            if j < len(tr):
                a.imshow(tr[j, 0], cmap='hot', vmin=vmin, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_ylabel(f'$S_{row+1}$ train', fontsize=9)
    fig.suptitle('Training sets (intensity) — each partition is one EIS scan (15 patches)', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/trainset_int.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {OUT_DIR}/trainset_int.png')


# ── figure: memorization grid at final checkpoint (shared color scale) ─────────
ROWS = ['Closest $S_1$ train', '$S_1$ generated', '$S_2$ generated', 'Closest $S_2$ train']


def fig_memorization(channel):
    ci, cmap = CH_IDX[channel], CMAPS[channel]
    rep = [final['near1'][:N_SHOW], final['s1'][:N_SHOW],
           final['s2'][:N_SHOW], final['near2'][:N_SHOW]]
    allvals = np.concatenate([r[:, ci].ravel() for r in rep])
    vmin, vmax = np.percentile(allvals, 1), np.percentile(allvals, 99)

    fig, ax = plt.subplots(4, N_SHOW, figsize=(1.6 * N_SHOW, 6.6))
    for r in range(4):
        for c in range(N_SHOW):
            a = ax[r, c]
            a.imshow(rep[r][c][ci], cmap=cmap, vmin=vmin, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_ylabel(ROWS[r], fontsize=8)
    fig.suptitle(f'Final available checkpoint ({steps[-1]:,} steps) — {channel} channel, shared color scale.\n'
                 f'Memorization ⇒ generated rows match the closest-train rows.', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/memorization_grid_{channel}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {OUT_DIR}/memorization_grid_{channel}.png')


# ── figure: convergence trajectory ────────────────────────────────────────────
def fig_convergence():
    st = np.array([t['step'] for t in traj])
    pair_m = np.array([t['pair_cos'].mean() for t in traj])
    pair_s = np.array([t['pair_cos'].std() for t in traj])
    clo = [np.concatenate([t['closest_cos_s1'], t['closest_cos_s2']]) for t in traj]
    clo_m = np.array([c.mean() for c in clo])
    clo_s = np.array([c.std() for c in clo])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(st, pair_m, '-o', color='tab:blue', label='pair cos ($S_1$ vs $S_2$) — model variance')
    ax.fill_between(st, pair_m - pair_s, pair_m + pair_s, color='tab:blue', alpha=0.15)
    ax.plot(st, clo_m, '-s', color='tab:orange', label='closest-train cos — memorization')
    ax.fill_between(st, clo_m - clo_s, clo_m + clo_s, color='tab:orange', alpha=0.15)
    ax.axhline(0, color='k', lw=0.6, alpha=0.4)
    ax.set_xlabel('training steps'); ax.set_ylabel('cosine similarity (global_logz, 3-ch)')
    ax.set_title('N=15 (1/2048) — emergence of memorization vs generalization')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/convergence.png', dpi=150)
    plt.close(fig)
    print(f'  saved {OUT_DIR}/convergence.png')


print('\nBuilding figures...')
fig_trainset()
for ch in ('int', 'vel', 'width'):
    fig_memorization(ch)
fig_convergence()

# ── save results + summary ────────────────────────────────────────────────────
np.save(f'{OUT_DIR}/results.npy', dict(
    config=dict(RUN_S1=RUN_S1, RUN_S2=RUN_S2, NUM_PAIRS=NUM_PAIRS,
                SAMPLING_TIMESTEPS=SAMPLING_TIMESTEPS, SEED=SEED, USE_EMA=USE_EMA,
                n_train_s1=len(tr1), n_train_s2=len(tr2)),
    trajectory=traj))

lines = [f'memorization2048 | N(S1)={len(tr1)} N(S2)={len(tr2)} | {NUM_PAIRS} pairs/milestone '
         f'| {SAMPLING_TIMESTEPS} DDIM steps | EMA={USE_EMA}', '',
         f'{"step":>7} {"mean pair cos":>15} {"mean closest cos":>18}', '-' * 42]
for t in traj:
    cc = 0.5 * (t['closest_cos_s1'].mean() + t['closest_cos_s2'].mean())
    lines.append(f'{t["step"]:>7} {t["pair_cos"].mean():>15.3f} {cc:>18.3f}')
summary = '\n'.join(lines)
print('\n' + summary)
with open(f'{OUT_DIR}/summary.txt', 'w') as f:
    f.write(summary + '\n')
print(f'\n  saved {OUT_DIR}/results.npy and {OUT_DIR}/summary.txt')