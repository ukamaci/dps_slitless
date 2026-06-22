"""
Memorization → generalization experiment — round 2: matched-compute.

Same protocol as ../analyze.py (Kadkhodaie et al., ICLR 2024, Fig. 2) but all
runs trained with matched compute (12,500 optimizer steps via train_num_steps):

  1/2    — original 140-epoch runs (same compute scale as the full-dataset run)
  1/8    — conv12500 (12.5k steps, model-final.pt)
  1/32   — conv12500 (12.5k steps, model-final.pt)
  1/128  — conv12500 (12.5k steps, model-final.pt)
  1/512  — conv12500 (12.5k steps, model-final.pt)
  1/2048 — mem25k runs, milestone 5 (= 12.5k steps, so matched to the rest)

Outputs (experiments/generalization_memorization/v2/outputs/):
  generalization_memorization_{int,vel,width}.png
  results.npy
  summary.txt

Run: python experiments/generalization_memorization/v2/analyze.py
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO_ROOT)
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import partition_files

# ── runs (hardcoded — experiment record) ─────────────────────────────────────
# Each entry: folder path (required) + optional 'checkpoint' override filename.
# Without 'checkpoint', find_checkpoint() is used (prefers model-final.pt then
# highest model-N.pt). The 1/2048 mem25k runs get model-5.pt to match 12.5k steps.
TR = os.path.join(REPO_ROOT, 'training_results')
RUNS = [
    # ── 1/2 (N≈11400): original 140-epoch schedule, model-10 is final ──────
    {'folder': f'{TR}/2026_06_01__06_12_41_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_None_dsize_1v2'},
    {'folder': f'{TR}/2026_06_01__10_38_13_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_None_dsize_2v2'},
    # ── 1/8 (N≈2851): conv12500 ─────────────────────────────────────────────
    {'folder': f'{TR}/2026_06_04__01_27_42_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_1v8_conv12500'},
    {'folder': f'{TR}/2026_06_04__03_46_03_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_2v8_conv12500'},
    # ── 1/32 (N≈710): conv12500 ─────────────────────────────────────────────
    {'folder': f'{TR}/2026_06_03__20_41_57_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_1v32_conv12500'},
    {'folder': f'{TR}/2026_06_03__23_04_35_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_2v32_conv12500'},
    # ── 1/128 (N≈176): conv12500 ────────────────────────────────────────────
    {'folder': f'{TR}/2026_06_04__02_19_40_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_1v128_conv12500'},
    {'folder': f'{TR}/2026_06_04__07_14_13_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_2v128_conv12500'},
    # ── 1/512 (N≈45): conv12500 ─────────────────────────────────────────────
    {'folder': f'{TR}/2026_06_03__19_54_14_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_1v512_conv12500'},
    {'folder': f'{TR}/2026_06_03__23_00_12_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_2v512_conv12500'},
    # ── 1/2048 (N=15): mem25k, milestone 5 = 12.5k steps (matched compute) ─
    {'folder': f'{TR}/2026_06_03__03_22_56_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_1v2048_mem25k',
     'checkpoint': 'model-5.pt'},
    {'folder': f'{TR}/2026_06_03__12_27_10_all_lr_1e-4_cosine_b32_numdetectors_0_global_logz_unconditional_Gaussian_30_dsize_2v2048_mem25k',
     'checkpoint': 'model-5.pt'},
]
OUT_DIR = os.path.join(REPO_ROOT, 'experiments/generalization_memorization/v2/outputs')

# ── config ──────────────────────────────────────────────────────────────────
NUM_PAIRS          = 30
SAMPLING_TIMESTEPS = 250
SEED               = 0
USE_EMA            = True
CLOSEST_SUBSAMPLE  = None
HIST_BINS          = 40
# ────────────────────────────────────────────────────────────────────────────

CMAPS   = {'int': 'hot', 'vel': 'seismic', 'width': 'plasma'}
CH_IDX  = {'int': 0, 'vel': 1, 'width': 2}
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DSHAPE  = (3, 64, 64)


def read_config(run_folder):
    with open(os.path.join(run_folder, 'config.json')) as f:
        return json.load(f)


def find_checkpoint(run_folder):
    """Prefer model-final.pt, else highest-numbered model-N.pt."""
    final = os.path.join(run_folder, 'model-final.pt')
    if os.path.exists(final):
        return final
    nums = [int(os.path.basename(p)[6:-3])
            for p in glob.glob(os.path.join(run_folder, 'model-*.pt'))
            if os.path.basename(p)[6:-3].isdigit()]
    if not nums:
        raise FileNotFoundError(f'no model-*.pt in {run_folder}')
    return os.path.join(run_folder, f'model-{max(nums)}.pt')


def resolve_checkpoint(run_entry):
    """Return full checkpoint path from a RUNS entry dict."""
    folder = run_entry['folder']
    if 'checkpoint' in run_entry:
        return os.path.join(folder, run_entry['checkpoint'])
    return find_checkpoint(folder)


def load_unet(ckpt_path, use_ema):
    model = Unet(channels=3, dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if use_ema:
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
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        s = diffusion.sample(batch_size=n)
    return s.detach().cpu().numpy()


def load_partition_images(dataset_path, partno, partnum, subsample=None):
    files = partition_files(glob.glob(dataset_path + '/data*.npy'), partno, partnum)
    if subsample is not None and subsample < len(files):
        files = list(np.array(files)[np.linspace(0, len(files) - 1, subsample).astype(int)])
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

# group runs into columns by partnum
by_partnum = {}
for entry in RUNS:
    cfg = read_config(entry['folder'])
    by_partnum.setdefault(cfg['partnum'], {})[cfg['partno']] = entry
partnums = sorted(by_partnum, reverse=True)   # descending partnum = ascending N = left→right
ncol = len(partnums)

columns = []
for c, pn in enumerate(partnums):
    entry1, entry2 = by_partnum[pn][1], by_partnum[pn][2]
    cfg1, cfg2 = read_config(entry1['folder']), read_config(entry2['folder'])
    ckpt1, ckpt2 = resolve_checkpoint(entry1), resolve_checkpoint(entry2)

    print(f'\n=== partition 1/{pn} (column {c + 1}/{ncol}) ===')
    print(f'  S1: {os.path.basename(ckpt1)}  |  S2: {os.path.basename(ckpt2)}')
    print('  sampling S1, S2 (shared init noise)...')

    diff1 = build_diffusion(load_unet(ckpt1, USE_EMA), cfg1)
    s1 = sample_model(diff1, NUM_PAIRS, SEED + c)
    del diff1
    diff2 = build_diffusion(load_unet(ckpt2, USE_EMA), cfg2)
    s2 = sample_model(diff2, NUM_PAIRS, SEED + c)
    del diff2
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    s1v, s2v = norm_flat(norm, s1), norm_flat(norm, s2)
    pair_cos = (s1v * s2v).sum(axis=1)

    print('  loading training partitions + closest-image search...')
    tr1 = load_partition_images(cfg1['dataset_path'], 1, pn, CLOSEST_SUBSAMPLE)
    tr2 = load_partition_images(cfg2['dataset_path'], 2, pn, CLOSEST_SUBSAMPLE)
    N = (len(tr1) + len(tr2)) // 2
    cclose1, arg1 = closest(s1v, norm_flat(norm, tr1))
    cclose2, arg2 = closest(s2v, norm_flat(norm, tr2))

    columns.append(dict(
        partnum=pn, N=N,
        s1=s1, s2=s2,
        closest_s1=tr1[arg1[0]], closest_s2=tr2[arg2[0]],
        pair_cos=pair_cos,
        closest_cos_s1=cclose1, closest_cos_s2=cclose2,
        ckpt1=ckpt1, ckpt2=ckpt2,
    ))
    cc = 0.5 * (cclose1.mean() + cclose2.mean())
    print(f'  N≈{N}  mean pair cos={pair_cos.mean():.3f}  mean closest cos={cc:.3f}')
    del tr1, tr2


# ── figures ──────────────────────────────────────────────────────────────────
ROW_LABELS = ['Closest image from $S_1$', 'Generated by model trained on $S_1$',
              'Generated by model trained on $S_2$', 'Closest image from $S_2$']


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
                ax.set_title(f'$N$≈{col["N"]}\n(1/{col["partnum"]})', fontsize=9)
            if c == 0:
                ax.set_ylabel(ROW_LABELS[r], fontsize=7)

        ax = fig.add_subplot(gs[4, c])
        orange = np.concatenate([col['closest_cos_s1'], col['closest_cos_s2']])
        ax.hist(col['pair_cos'], bins=HIST_BINS, range=(-1, 1), color='tab:blue',
                alpha=0.7, label='pair ($S_1$ vs $S_2$)')
        ax.hist(orange, bins=HIST_BINS, range=(-1, 1), color='tab:orange',
                alpha=0.7, label='closest train image')
        ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
        ax.set_xlim(-1, 1); ax.set_yticks([])
        ax.set_xlabel('cosine similarity', fontsize=8)
        if c == 0:
            ax.legend(fontsize=6, loc='upper left')

    fig.suptitle(f'Memorization → generalization  (matched compute, 12.5k steps) '
                 f'— {channel} channel shown; cosine over all 3 ch in global_logz space',
                 fontsize=10)
    out = f'{OUT_DIR}/generalization_memorization_{channel}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


print('\nBuilding figures...')
for ch in ('int', 'vel', 'width'):
    make_figure(ch)


# ── save results ─────────────────────────────────────────────────────────────
results = dict(
    config=dict(NUM_PAIRS=NUM_PAIRS, SAMPLING_TIMESTEPS=SAMPLING_TIMESTEPS,
                SEED=SEED, USE_EMA=USE_EMA, CLOSEST_SUBSAMPLE=CLOSEST_SUBSAMPLE,
                runs=[r['folder'] for r in RUNS]),
    columns=[{k: v for k, v in col.items() if k not in ('s1', 's2')} for col in columns],
)
np.save(f'{OUT_DIR}/results.npy', results)

lines = [f'Memorization→generalization v2 | matched compute (12.5k steps) | '
         f'{NUM_PAIRS} pairs/col | {SAMPLING_TIMESTEPS} DDIM steps | EMA={USE_EMA}', '',
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
