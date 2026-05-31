"""
Conditional diffusion normalization experiment — analysis.

Reads outputs/results.npy produced by runner.py and generates:
  outputs/rmse_vs_milestone.png  — RMSE vs milestone, per channel, both runs
  outputs/rmse_table.txt         — mean ± std table for best milestone
  outputs/recon_grid_ms{N}.png   — reconstruction grids at selected milestones

Edit the config block below, then:
    python experiments/conddiff_normalization/analyze.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_FILE  = os.path.join(REPO_ROOT, 'experiments/conddiff_normalization/outputs/results.npy')
OUT_DIR       = os.path.join(REPO_ROOT, 'experiments/conddiff_normalization/outputs')
GRID_MS       = [1, 10]        # milestones to show reconstruction grids for
GRID_VAL_IDX  = 0              # which val image to use for reconstruction grids
# ─────────────────────────────────────────────────────────────────────────────

CH_LABELS = ['int (erg/cm²/s/sr)', 'vel (km/s)', 'width (km/s)']
CH_SHORT  = ['int', 'vel', 'width']
CMAPS     = ['hot', 'seismic', 'plasma']
RUN_COLORS = {'logz': '#1f77b4', 'linear': '#ff7f0e'}
RUN_LABELS = {'logz': 'global_logz', 'linear': 'persample_linear'}

os.makedirs(OUT_DIR, exist_ok=True)

r = np.load(RESULTS_FILE, allow_pickle=True).item()
cfg        = r['config']
milestones = cfg['milestones']
true_all   = r['true']   # (N_VAL, 3, H, W)
meas_all   = r['meas']   # (N_VAL, K, H, W)
run_names  = [k for k in r if k not in ('config', 'true', 'meas')]
N_VAL      = len(true_all)

# ── 1. RMSE vs milestone ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for run_name in run_names:
    rmse_mean = r[run_name]['rmse_mean']   # (N_MS, N_VAL, 3)
    mu  = rmse_mean.mean(axis=1)           # (N_MS, 3)
    std = rmse_mean.std(axis=1)            # (N_MS, 3)
    for c, ax in enumerate(axes):
        ax.errorbar(milestones, mu[:, c], yerr=std[:, c],
                    label=RUN_LABELS[run_name], color=RUN_COLORS[run_name],
                    marker='o', capsize=3)

for c, ax in enumerate(axes):
    ax.set_title(CH_SHORT[c]); ax.set_xlabel('Milestone')
    ax.set_xticks(milestones)
    ax.set_ylabel(CH_LABELS[c]); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

axes[0].set_ylabel('RMSE posterior mean')
fig.suptitle('Reconstruction RMSE vs milestone — conditional diffusion', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'rmse_vs_milestone.png'), dpi=150)
plt.close(fig)
print('Saved rmse_vs_milestone.png')


# ── 2. RMSE table (best milestone per run) ────────────────────────────────────
lines = []
lines.append(f'{"Run":<20} {"MS":>4}  ' + '  '.join(f'{l:>24}' for l in CH_SHORT))
lines.append('-' * 80)
for run_name in run_names:
    rmse_mean = r[run_name]['rmse_mean']            # (N_MS, N_VAL, 3)
    for ms_idx, ms in enumerate(milestones):
        mu  = rmse_mean[ms_idx].mean(axis=0)        # (3,)
        std = rmse_mean[ms_idx].std(axis=0)         # (3,)
        vals = '  '.join(f'{mu[c]:.2f} ± {std[c]:.2f}' for c in range(3))
        lines.append(f'{RUN_LABELS[run_name]:<20} {ms:>4}  {vals}')
    lines.append('')

table_path = os.path.join(OUT_DIR, 'rmse_table.txt')
with open(table_path, 'w') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
print(f'Saved rmse_table.txt')


# ── 3. Posterior mean RMSE box plot at last milestone ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
best_ms_idx = len(milestones) - 1
for c, ax in enumerate(axes):
    data, labels, colors = [], [], []
    for run_name in run_names:
        vals = r[run_name]['rmse_mean'][best_ms_idx, :, c]   # (N_VAL,)
        data.append(vals); labels.append(RUN_LABELS[run_name]); colors.append(RUN_COLORS[run_name])
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_title(CH_SHORT[c])
    ax.set_ylabel(CH_LABELS[c]); ax.grid(True, alpha=0.3)

fig.suptitle(f'Posterior mean RMSE across val images — milestone {milestones[best_ms_idx]}', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'rmse_boxplot.png'), dpi=150)
plt.close(fig)
print('Saved rmse_boxplot.png')


# ── 4. Reconstruction grids ───────────────────────────────────────────────────
vi  = GRID_VAL_IDX
true_np = true_all[vi]   # (3, H, W)
meas_np = meas_all[vi]   # (K, H, W)
vmins = [true_np[c].min() for c in range(3)]
vmaxs = [true_np[c].max() for c in range(3)]

for ms in GRID_MS:
    if ms not in milestones:
        continue
    ms_idx = milestones.index(ms)

    cols = [('Meas',  meas_np[:3])]   # first 3 measurement orders
    cols.append(('True', true_np))
    for run_name in run_names:
        samp  = r[run_name]['samples'][ms_idx, vi]   # (N_SAMPLES, 3, H, W)
        mean_s = samp.mean(axis=0)
        cols.append((f'{RUN_LABELS[run_name]}\nms={ms}', mean_s))

    fig, axes = plt.subplots(3, len(cols), figsize=(3 * len(cols), 8))
    for col_idx, (title, img) in enumerate(cols):
        is_meas = col_idx == 0
        for row in range(3):
            data = img[row] if img.shape[0] > row else img[0]
            cmap = 'hot' if is_meas else CMAPS[row]
            im = axes[row, col_idx].imshow(
                data, cmap=cmap,
                vmin=vmins[row] if not is_meas else None,
                vmax=vmaxs[row] if not is_meas else None,
            )
            axes[row, col_idx].axis('off')
            if row == 0:
                axes[row, col_idx].set_title(title, fontsize=8)
            if not is_meas:
                plt.colorbar(im, ax=axes[row, col_idx], fraction=0.046, pad=0.04)

    fig.suptitle(f'Reconstruction — val idx {vi}, ms={ms}', fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'recon_grid_ms{ms}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved recon_grid_ms{ms}.png')


# ── 5. RMSE heatmap: run × milestone (per channel) ───────────────────────────
for c in range(3):
    mat = np.stack([r[rn]['rmse_mean'][:, :, c].mean(axis=1) for rn in run_names])
    # mat: (N_RUNS, N_MS)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    im = ax.imshow(mat, aspect='auto', cmap='viridis_r')
    ax.set_xticks(range(len(milestones))); ax.set_xticklabels([str(m) for m in milestones])
    ax.set_yticks(range(len(run_names)));  ax.set_yticklabels([RUN_LABELS[rn] for rn in run_names])
    ax.set_xlabel('Milestone'); ax.set_title(f'Mean RMSE — {CH_SHORT[c]}')
    plt.colorbar(im, ax=ax, label=CH_LABELS[c])
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'rmse_heatmap_{CH_SHORT[c]}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved rmse_heatmap_{CH_SHORT[c]}.png')

print('\nDone.')