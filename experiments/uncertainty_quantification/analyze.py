"""
Uncertainty quantification experiment — analysis.

Reads outputs/results.npy produced by runner.py and generates:

  exp1 (in-distribution):
    outputs/exp1_uq_maps_example.png   — predictive mean / AU / EU / pred-var maps
    outputs/exp1_calibration.png       — calibration curves per channel
    outputs/exp1_rmse_comparison.png   — ensemble mean vs single-model RMSE
    outputs/exp1_summary.txt           — RMSE, AU/EU, calibration, sharpness, NLL, CRPS table

  exp2 (noise-level shift):
    outputs/exp2_au_eu_vs_dbsnr.png    — AU/EU vs measurement dB-SNR, per channel
    outputs/exp2_rmse_vs_dbsnr.png     — RMSE vs dB-SNR
    outputs/exp2_summary.txt

  exp3 (synthetic phantom):
    outputs/exp3_phantom_uq_maps.png   — AU/EU maps, phantom vs in-distribution example
    outputs/exp3_summary.txt

Run:
    python experiments/uncertainty_quantification/analyze.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

# ── config ────────────────────────────────────────────────────────────────────
RESULTS_FILE = os.path.join(REPO_ROOT, 'experiments/uncertainty_quantification/outputs/results.npy')
OUT_DIR      = os.path.join(REPO_ROOT, 'experiments/uncertainty_quantification/outputs')
EXAMPLE_IDX  = 0          # which val image to use for exp1 maps
N_CAL_BINS   = 100        # confidence levels for calibration curves
EPS          = 1e-8       # numerical floor for predictive variance
# ─────────────────────────────────────────────────────────────────────────────

CH_LABELS = ['int (erg/cm²/s/sr)', 'vel (km/s)', 'width (km/s)']
CH_SHORT  = ['int', 'vel', 'width']
CMAPS     = ['hot', 'seismic', 'plasma']

os.makedirs(OUT_DIR, exist_ok=True)
r   = np.load(RESULTS_FILE, allow_pickle=True).item()
cfg = r['config']
T1, T2 = cfg['T1'], cfg['T2']


# ── core decomposition ───────────────────────────────────────────────────────
def decompose(samples, t2_axis, t1_axis):
    """Decompose predictive uncertainty via the law of total variance.

    samples: array with a T2 axis (ensemble members) and a T1 axis (posterior
    samples per member), e.g. (..., T2, T1, C, H, W).

    Returns (pred_mean, AU, EU, pred_var), each with the T2/T1 axes removed.
      AU (aleatoric)  = E_models[ Var_samples ]   — within-model spread
      EU (epistemic)  = Var_models[ E_samples ]   — across-model disagreement
      pred_var        = AU + EU
    """
    per_model_mean = samples.mean(axis=t1_axis)            # T1 axis removed
    per_model_var  = samples.var(axis=t1_axis)             # T1 axis removed
    # t2_axis position unchanged by removing t1_axis only if t1_axis > t2_axis
    AU = per_model_var.mean(axis=t2_axis)
    EU = per_model_mean.var(axis=t2_axis)
    pred_mean = per_model_mean.mean(axis=t2_axis)
    pred_var  = AU + EU
    return pred_mean, AU, EU, pred_var, per_model_mean


def rmse_ch(a, b, axes=(-2, -1)):
    return np.sqrt(np.mean((a - b) ** 2, axis=axes))


def calibration_curve(true, mean, var, n_bins=N_CAL_BINS):
    """Returns (p_nominal, p_empirical) over flattened pixels."""
    sigma = np.sqrt(np.clip(var, EPS, None))
    z = (true - mean) / sigma
    cdf = norm.cdf(z).ravel()
    p_nominal = np.linspace(0, 1, n_bins)
    p_empirical = np.array([(cdf <= p).mean() for p in p_nominal])
    return p_nominal, p_empirical


def calibration_metrics(p_nom, p_emp):
    mace  = np.mean(np.abs(p_emp - p_nom))
    rmsce = np.sqrt(np.mean((p_emp - p_nom) ** 2))
    ma    = np.trapz(np.abs(p_emp - p_nom), p_nom)
    return mace, rmsce, ma


def nll_gaussian(true, mean, var):
    var = np.clip(var, EPS, None)
    return np.mean(0.5 * np.log(2 * np.pi * var) + (true - mean) ** 2 / (2 * var))


def crps_gaussian(true, mean, var):
    """Standard (positively-oriented, lower=better) closed-form CRPS for a
    Gaussian predictive distribution (Gneiting & Raftery 2007)."""
    sigma = np.sqrt(np.clip(var, EPS, None))
    z = (true - mean) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.mean(crps)


def sharpness(var):
    return np.sqrt(np.mean(np.clip(var, 0, None)))


# ════════════════════════════════════════════════════════════════════════════
# Exp 1 — in-distribution AU/EU decomposition + reconstruction quality + calibration
# ════════════════════════════════════════════════════════════════════════════
print('── Exp 1: in-distribution ──')
e1 = r['exp1']
true1    = e1['true']                                  # (N_VAL,3,H,W)
samples1 = e1['samples']                               # (N_VAL,T2,T1,3,H,W)
N_VAL = true1.shape[0]

pred_mean1, AU1, EU1, predvar1, per_model_mean1 = decompose(samples1, t2_axis=1, t1_axis=2)
# pred_mean1, AU1, EU1, predvar1: (N_VAL,3,H,W); per_model_mean1: (N_VAL,T2,3,H,W)

# -- RMSE: ensemble mean vs single-model means --
rmse_ensemble = rmse_ch(true1, pred_mean1)             # (N_VAL,3)
rmse_single   = np.stack([rmse_ch(true1, per_model_mean1[:, m]) for m in range(T2)])  # (T2,N_VAL,3)

lines = []
lines.append('Exp 1 — in-distribution (N_VAL=%d, T1=%d, T2=%d)\n' % (N_VAL, T1, T2))
lines.append(f'{"Method":<22}' + ''.join(f'{c:>22}' for c in CH_SHORT))
lines.append('-' * (22 + 22 * 3))
mu, sd = rmse_ensemble.mean(0), rmse_ensemble.std(0)
lines.append(f'{"Ensemble mean":<22}' + ''.join(f'{mu[c]:>10.3f} ± {sd[c]:<7.3f}' for c in range(3)))
for m in range(T2):
    mu, sd = rmse_single[m].mean(0), rmse_single[m].std(0)
    lines.append(f'{"Model " + str(m+1):<22}' + ''.join(f'{mu[c]:>10.3f} ± {sd[c]:<7.3f}' for c in range(3)))
lines.append('')

# -- AU / EU summary (averaged over images and pixels) --
au_mean = AU1.mean(axis=(0, 2, 3))     # (3,)
eu_mean = EU1.mean(axis=(0, 2, 3))
pv_mean = predvar1.mean(axis=(0, 2, 3))
lines.append(f'{"Uncertainty (sqrt, mean over px/img)":<38}' + ''.join(f'{c:>16}' for c in CH_SHORT))
lines.append('-' * (38 + 16 * 3))
lines.append(f'{"AU (aleatoric)":<38}' + ''.join(f'{np.sqrt(au_mean[c]):>16.4g}' for c in range(3)))
lines.append(f'{"EU (epistemic)":<38}' + ''.join(f'{np.sqrt(eu_mean[c]):>16.4g}' for c in range(3)))
lines.append(f'{"Predictive (AU+EU)":<38}' + ''.join(f'{np.sqrt(pv_mean[c]):>16.4g}' for c in range(3)))
lines.append(f'{"EU / Predictive (fraction)":<38}' + ''.join(f'{(eu_mean[c]/pv_mean[c]):>16.4f}' for c in range(3)))
lines.append('')

# -- calibration / sharpness / scoring rules (per channel, pooled over images+pixels) --
lines.append(f'{"Calibration & scoring":<22}' + ''.join(f'{c:>16}' for c in CH_SHORT))
lines.append('-' * (22 + 16 * 3))
cal_curves = {}
mace_l, rmsce_l, ma_l, sharp_l, nll_l, crps_l = [], [], [], [], [], []
for c in range(3):
    p_nom, p_emp = calibration_curve(true1[:, c], pred_mean1[:, c], predvar1[:, c])
    cal_curves[c] = (p_nom, p_emp)
    mace, rmsce, ma = calibration_metrics(p_nom, p_emp)
    sh   = sharpness(predvar1[:, c])
    nll  = nll_gaussian(true1[:, c], pred_mean1[:, c], predvar1[:, c])
    crps = crps_gaussian(true1[:, c], pred_mean1[:, c], predvar1[:, c])
    mace_l.append(mace); rmsce_l.append(rmsce); ma_l.append(ma)
    sharp_l.append(sh); nll_l.append(nll); crps_l.append(crps)

lines.append(f'{"RMSCE":<22}' + ''.join(f'{v:>16.4f}' for v in rmsce_l))
lines.append(f'{"MACE":<22}'  + ''.join(f'{v:>16.4f}' for v in mace_l))
lines.append(f'{"MA":<22}'    + ''.join(f'{v:>16.4f}' for v in ma_l))
lines.append(f'{"Sharpness":<22}' + ''.join(f'{v:>16.4g}' for v in sharp_l))
lines.append(f'{"NLL":<22}'   + ''.join(f'{v:>16.4g}' for v in nll_l))
lines.append(f'{"CRPS":<22}'  + ''.join(f'{v:>16.4g}' for v in crps_l))
lines.append('')

with open(os.path.join(OUT_DIR, 'exp1_summary.txt'), 'w') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
print('Saved exp1_summary.txt')

# -- figure: AU/EU/predvar maps for an example image --
vi = EXAMPLE_IDX
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
col_titles = ['Predictive mean', 'AU (aleatoric, std)', 'EU (epistemic, std)', 'Pred. uncertainty (std)']
for c in range(3):
    maps = [pred_mean1[vi, c], np.sqrt(AU1[vi, c]), np.sqrt(EU1[vi, c]), np.sqrt(predvar1[vi, c])]
    cmaps_row = [CMAPS[c], 'viridis', 'viridis', 'viridis']
    for j, (m, cm) in enumerate(zip(maps, cmaps_row)):
        im = axes[c, j].imshow(m, cmap=cm)
        axes[c, j].axis('off')
        if c == 0:
            axes[c, j].set_title(col_titles[j], fontsize=10)
        if j == 0:
            axes[c, j].text(-0.15, 0.5, CH_SHORT[c], transform=axes[c, j].transAxes,
                             fontsize=11, va='center', ha='right', rotation=90)
        plt.colorbar(im, ax=axes[c, j], fraction=0.046, pad=0.04)
fig.suptitle(f'Exp 1 — UQ decomposition (val idx {vi})', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exp1_uq_maps_example.png'), dpi=150)
plt.close(fig)
print('Saved exp1_uq_maps_example.png')

# -- figure: calibration curves --
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for c, ax in enumerate(axes):
    p_nom, p_emp = cal_curves[c]
    ax.plot(p_nom, p_emp, label='Ensemble (proposed)', color='tab:blue')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    ax.set_title(CH_SHORT[c]); ax.set_xlabel('Nominal confidence')
    ax.set_ylabel('Empirical confidence'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
fig.suptitle('Exp 1 — calibration curves', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exp1_calibration.png'), dpi=150)
plt.close(fig)
print('Saved exp1_calibration.png')

# -- figure: RMSE comparison ensemble vs single models --
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for c, ax in enumerate(axes):
    data = [rmse_ensemble[:, c]] + [rmse_single[m, :, c] for m in range(T2)]
    labels = ['Ensemble'] + [f'M{m+1}' for m in range(T2)]
    colors = ['tab:blue'] + ['tab:gray'] * T2
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_title(CH_SHORT[c]); ax.set_ylabel(CH_LABELS[c]); ax.grid(True, alpha=0.3)
fig.suptitle('Exp 1 — posterior-mean RMSE: ensemble vs individual models', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exp1_rmse_comparison.png'), dpi=150)
plt.close(fig)
print('Saved exp1_rmse_comparison.png')


# ════════════════════════════════════════════════════════════════════════════
# Exp 2 — OOD via measurement-noise-level shift
# ════════════════════════════════════════════════════════════════════════════
print('\n── Exp 2: noise-level shift ──')
e2 = r['exp2']
true2    = e2['true']                  # (N_VAL2,3,H,W)
samples2 = e2['samples']               # (n_dbsnr,N_VAL2,T2,T1,3,H,W)
dbsnr_levels = cfg['dbsnr_levels']
train_dbsnr  = cfg['train_dbsnr']
n_dbsnr, N_VAL2 = samples2.shape[:2]

au2 = np.zeros((n_dbsnr, 3)); eu2 = np.zeros((n_dbsnr, 3)); pv2 = np.zeros((n_dbsnr, 3))
rmse2 = np.zeros((n_dbsnr, N_VAL2, 3))
for di in range(n_dbsnr):
    pm, au, eu, pv, _ = decompose(samples2[di], t2_axis=1, t1_axis=2)  # axes within (N_VAL2,T2,T1,3,H,W)
    au2[di] = au.mean(axis=(0, 2, 3))
    eu2[di] = eu.mean(axis=(0, 2, 3))
    pv2[di] = pv.mean(axis=(0, 2, 3))
    rmse2[di] = rmse_ch(true2, pm)

lines = []
lines.append('Exp 2 — measurement-noise-level shift (N_VAL=%d, train dbsnr=%d)\n' % (N_VAL2, train_dbsnr))
lines.append(f'{"dbsnr":>8}' + ''.join(f'{"AU_"+c:>14}{"EU_"+c:>14}{"RMSE_"+c:>14}' for c in CH_SHORT))
for di, dbsnr in enumerate(dbsnr_levels):
    row = f'{dbsnr:>8}'
    for c in range(3):
        row += f'{np.sqrt(au2[di,c]):>14.4g}{np.sqrt(eu2[di,c]):>14.4g}{rmse2[di,:,c].mean():>14.4g}'
    lines.append(row)
with open(os.path.join(OUT_DIR, 'exp2_summary.txt'), 'w') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
print('Saved exp2_summary.txt')

# -- figure: AU/EU vs dbsnr --
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for c, ax in enumerate(axes):
    ax.plot(dbsnr_levels, np.sqrt(au2[:, c]), 'o-', label='AU (aleatoric)', color='tab:orange')
    ax.plot(dbsnr_levels, np.sqrt(eu2[:, c]), 's-', label='EU (epistemic)', color='tab:red')
    ax.axvline(train_dbsnr, ls='--', color='gray', label='Training dbsnr')
    ax.set_title(CH_SHORT[c]); ax.set_xlabel('Measurement dB-SNR')
    ax.set_ylabel(CH_LABELS[c] + ' (std)'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.invert_xaxis()
fig.suptitle('Exp 2 — AU/EU vs measurement noise level', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exp2_au_eu_vs_dbsnr.png'), dpi=150)
plt.close(fig)
print('Saved exp2_au_eu_vs_dbsnr.png')

# -- figure: RMSE vs dbsnr --
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for c, ax in enumerate(axes):
    mu  = rmse2[:, :, c].mean(axis=1)
    std = rmse2[:, :, c].std(axis=1)
    ax.errorbar(dbsnr_levels, mu, yerr=std, marker='o', capsize=3, color='tab:blue')
    ax.axvline(train_dbsnr, ls='--', color='gray', label='Training dbsnr')
    ax.set_title(CH_SHORT[c]); ax.set_xlabel('Measurement dB-SNR')
    ax.set_ylabel(CH_LABELS[c]); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.invert_xaxis()
fig.suptitle('Exp 2 — reconstruction RMSE vs measurement noise level', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exp2_rmse_vs_dbsnr.png'), dpi=150)
plt.close(fig)
print('Saved exp2_rmse_vs_dbsnr.png')


# ════════════════════════════════════════════════════════════════════════════
# Exp 3 — OOD via synthetic phantom
# ════════════════════════════════════════════════════════════════════════════
print('\n── Exp 3: synthetic phantom ──')
e3 = r['exp3']
kinds     = e3['kinds']
true3     = e3['true']                  # (n_phantom,3,H,W)
samples3  = e3['samples']               # (n_phantom,T2,T1,3,H,W)
n_phantom = len(kinds)

pred_mean3, AU3, EU3, predvar3, _ = decompose(samples3, t2_axis=1, t1_axis=2)

# in-distribution reference: average AU/EU over exp1 val set
au1_mean = AU1.mean(axis=(0, 2, 3))
eu1_mean = EU1.mean(axis=(0, 2, 3))

lines = []
lines.append('Exp 3 — synthetic phantom (sqrt of mean AU/EU over pixels)\n')
lines.append(f'{"Source":<22}' + ''.join(f'{"AU_"+c:>14}{"EU_"+c:>14}' for c in CH_SHORT))
lines.append(f'{"Exp1 in-distribution":<22}' + ''.join(f'{np.sqrt(au1_mean[c]):>14.4g}{np.sqrt(eu1_mean[c]):>14.4g}' for c in range(3)))
for pi, kind in enumerate(kinds):
    au_p = AU3[pi].mean(axis=(-2, -1))
    eu_p = EU3[pi].mean(axis=(-2, -1))
    lines.append(f'{kind:<22}' + ''.join(f'{np.sqrt(au_p[c]):>14.4g}{np.sqrt(eu_p[c]):>14.4g}' for c in range(3)))
lines.append('')
lines.append('EU ratio (phantom / in-distribution):')
for pi, kind in enumerate(kinds):
    eu_p = EU3[pi].mean(axis=(-2, -1))
    ratio = eu_p / eu1_mean
    lines.append(f'{kind:<22}' + ''.join(f'{"EU_"+CH_SHORT[c]+"_ratio":>14}: {ratio[c]:.2f}  ' for c in range(3)))

with open(os.path.join(OUT_DIR, 'exp3_summary.txt'), 'w') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
print('Saved exp3_summary.txt')

# -- figure: phantom true / pred-mean / AU / EU, one row per phantom + in-dist example --
n_rows = n_phantom + 1
fig, axes = plt.subplots(n_rows, 3 * 4, figsize=(3 * 4 * 3, 3 * n_rows))
if n_rows == 1:
    axes = axes[None, :]

def plot_row(ax_row, true_img, mean_img, au_img, eu_img, row_label):
    for c in range(3):
        base = c * 4
        panels = [(true_img[c], CMAPS[c], 'True'), (mean_img[c], CMAPS[c], 'Pred. mean'),
                  (np.sqrt(au_img[c]), 'viridis', 'AU (std)'), (np.sqrt(eu_img[c]), 'viridis', 'EU (std)')]
        for j, (img, cm, title) in enumerate(panels):
            ax = ax_row[base + j]
            im = ax.imshow(img, cmap=cm)
            ax.axis('off')
            ax.set_title(f'{CH_SHORT[c]} {title}', fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax_row[0].text(-0.3, 0.5, row_label, transform=ax_row[0].transAxes,
                    fontsize=11, va='center', ha='right', rotation=90)

# in-distribution example (reuse exp1 val image EXAMPLE_IDX)
plot_row(axes[0], true1[EXAMPLE_IDX], pred_mean1[EXAMPLE_IDX], AU1[EXAMPLE_IDX], EU1[EXAMPLE_IDX], 'In-dist (val)')
for pi, kind in enumerate(kinds):
    plot_row(axes[pi + 1], true3[pi], pred_mean3[pi], AU3[pi], EU3[pi], f'Phantom: {kind}')

fig.suptitle('Exp 3 — AU/EU maps: in-distribution vs synthetic phantoms', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exp3_phantom_uq_maps.png'), dpi=130)
plt.close(fig)
print('Saved exp3_phantom_uq_maps.png')

print('\nDone.')
