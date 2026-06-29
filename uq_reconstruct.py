"""
uq_reconstruct.py — deep-ensemble aleatoric + epistemic uncertainty for
conditional-diffusion reconstruction (analogue of reconstruct.py).

Ensemble = the 8 `*_conditional_Gaussian_20` models. Each is a conditional
diffusion model that, given a (noisy) measurement y, samples from p(x | y).
For one fixed measurement we draw N posterior samples per member and apply the
law of total variance:

    Var(x | y) = E_m[ Var(x | y, member m) ]   +   Var_m[ E(x | y, member m) ]
               = aleatoric (avg within-model spread) + epistemic (model disagree)

    posterior mean = mean_m E(x | y, member m)

Uncertainty maps are shown as std (= sqrt of the variances), in physical units.

Figure (3×5): rows = channels (int / vel / width)
    col 1 True | col 2 Posterior Mean | col 3 Pred. Uncertainty | col 4 Aleatoric | col 5 Epistemic
Brackets: posterior-mean column = per-channel RMSE; uncertainty columns = spatial-mean std.
"""
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from slitless.forward import Imager, Source
from slitless.recon import conddiff_solver

# ── config ──────────────────────────────────────────────────────────────────────
SAMPLE_IDX  = 20               # test50 image index
NUM_SAMPLES = 10              # posterior samples per ensemble member
NOISE       = (20, 'gaussian')  # measurement noise; matches the Gaussian_20 models
SAVE        = True
SUPTITLE    = False            # show the figure suptitle
SHOW_METRICS = True           # draw per-panel RMSE/MAE metric labels inside the panels

ENSEMBLE_GLOB = 'training_results/*conditional_Gaussian_20'
TEST50_PATH   = '/home/kamo/resources/slitless/data/datasets/baseline/eis_test_50_dsetv6.npy'
OUTPUT_DIR    = 'experiments/uncertainty_quantification/outputs'

SPEED_OF_LIGHT  = 299792.458
REST_WL         = 195.117937907451
SPECTRAL_ORDERS = [0, -1, 1]

CMAPS      = ['hot', 'seismic', 'plasma']
UNC_CMAP   = 'jet'
ROW_LABELS = ['Intensity\n[erg/cm²/s/sr]', 'Velocity\n[km/s]', 'Line Width\n[km/s]']
COL_TITLES = ['True', 'Posterior Mean', 'Posterior Mean Error', 'Total Uncertainty',
              'Aleatoric', 'Epistemic']


def find_ckpt(folder):
    """Pick the checkpoint for an ensemble member: model-final.pt, else
    model-10.pt, else the highest model-N.pt."""
    for name in ('model-final.pt', 'model-10.pt'):
        if os.path.exists(os.path.join(folder, name)):
            return name
    cands = []
    for c in glob.glob(os.path.join(folder, 'model-*.pt')):
        m = re.search(r'model-(\d+)\.pt$', c)
        if m:
            cands.append((int(m.group(1)), os.path.basename(c)))
    if not cands:
        raise FileNotFoundError(f'no model-*.pt in {folder}')
    return max(cands)[1]


def fmt(row, v):
    return f'{v:.1f}' if row == 0 else f'{v:.2f}'


def rmse_ch(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def panel_label(ax, text):
    """Draw a metric label inside the panel (mid-bottom), large and readable on
    any colormap via white text with a black outline."""
    t = ax.text(0.5, 0.03, text, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=16, fontweight='bold', color='white')
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground='black')])


# ── data + ensemble ──────────────────────────────────────────────────────────────
data      = np.load(TEST50_PATH, allow_pickle=True).item()
param4dar = data['param3d'].astype(np.float32)        # (50, 3, 64, 64) physical, width Å

members = sorted(glob.glob(ENSEMBLE_GLOB))
assert members, f'no ensemble members matched {ENSEMBLE_GLOB}'
print(f'Ensemble: {len(members)} members')

# ── measurement (one fixed noisy realization, shared across the ensemble) ────────
dbsnr, noise_model = NOISE
Imgr = Imager(
    pixelated=True,
    spectral_orders=SPECTRAL_ORDERS,
    dispersion_scale=0.022275,
    mid_wavelength=195.119,
    dbsnr=dbsnr,
    noise_model=noise_model,
)
src = Source(param3d=param4dar[SAMPLE_IDX].copy(), pix=False)
Imgr.get_measurements(sources=src, no_noise=(dbsnr is None))   # sets meas3dar + srpix

# ── sample every member on that fixed measurement ────────────────────────────────
samples_pix = []   # per member: (N, 3, 64, 64) pixel units
for i, folder in enumerate(members):
    ckpt = find_ckpt(folder)
    print(f'  [{i+1}/{len(members)}] {os.path.basename(folder)}  ({ckpt})')
    _, _, samp = conddiff_solver(
        imager=Imgr,
        run_name=os.path.basename(folder),
        model_path=ckpt,
        num_samples=NUM_SAMPLES,
        return_samples=True,
    )
    samples_pix.append(samp)

samples_pix = np.stack(samples_pix)                            # (M, N, 3, 64, 64)

# ── to physical units (width km/s), then decompose ──────────────────────────────
samples = Imgr.frompix(samples_pix, width_unit='km/s', array=True)   # (M, N, 3, 64, 64)

mu_m     = samples.mean(axis=1)        # (M, 3, 64, 64)  per-member predictive mean
v_m      = samples.var(axis=1)         # (M, 3, 64, 64)  within-member variance

post_mean = mu_m.mean(axis=0)          # (3, 64, 64)
ale_var   = v_m.mean(axis=0)           # aleatoric  = mean within-member variance
epi_var   = mu_m.var(axis=0)           # epistemic  = variance of member means
pred_var  = ale_var + epi_var          # total predictive variance

ale_std, epi_std, pred_std = np.sqrt(ale_var), np.sqrt(epi_var), np.sqrt(pred_var)

true = param4dar[SAMPLE_IDX].copy()
true[2] *= SPEED_OF_LIGHT / REST_WL    # width Å → km/s

# ── console summary ──────────────────────────────────────────────────────────────
print(f'\nidx {SAMPLE_IDX}  (int erg/cm²/s/sr, vel km/s, width km/s)')
for r, name in enumerate(['int  ', 'vel  ', 'width']):
    print(f'  {name}  RMSE={rmse_ch(post_mean[r], true[r]):8.3f}   '
          f'ale={ale_std[r].mean():.3f}  epi={epi_std[r].mean():.3f}  pred={pred_std[r].mean():.3f}')

# ── figure ───────────────────────────────────────────────────────────────────────
# gridspec reserves thin columns (idx 2 and 7) for the two colorbar groups so the
# param colorbar after 'Posterior Mean' doesn't collide with the magnitude panels.
# Magnitude group (cols 2-5: Error, Total, Aleatoric, Epistemic) shares one scale per
# row (vmax = total-uncertainty max), so the error map is directly comparable to the
# predicted uncertainty — i.e. is the ensemble uncertain where it is actually wrong?
abs_err  = np.abs(post_mean - true)                   # (3, 64, 64)
unc_maps = [abs_err, pred_std, ale_std, epi_std]      # order matches COL_TITLES[2:]
fig = plt.figure(figsize=(14, 6.8))
gs  = fig.add_gridspec(3, 8, width_ratios=[1, 1, 0.2, 1, 1, 1, 1, 0.07],   # wide param-cb col → room for ticks before col-3
                       left=0.06, right=0.96, top=0.90, bottom=0.05,
                       hspace=0.10, wspace=0.08)
IMG_GC = [0, 1, 3, 4, 5, 6]   # gridspec columns that hold image panels (cols 0..5)

for r in range(3):
    vmin, vmax = true[r].min(), true[r].max()
    umax = pred_std[r].max()                          # shared magnitude scale per row
    param_m = unc_m = None
    for c, gc in enumerate(IMG_GC):
        a = fig.add_subplot(gs[r, gc])
        if c == 0:
            param_m = a.imshow(true[r], cmap=CMAPS[r], vmin=vmin, vmax=vmax)
        elif c == 1:
            a.imshow(post_mean[r], cmap=CMAPS[r], vmin=vmin, vmax=vmax)
            if SHOW_METRICS:
                panel_label(a, fmt(r, rmse_ch(post_mean[r], true[r])))
        else:
            umap = unc_maps[c - 2][r]
            unc_m = a.imshow(umap, cmap=UNC_CMAP, vmin=0, vmax=umax)
            if SHOW_METRICS:
                panel_label(a, fmt(r, umap.mean()))
        a.set_xticks([]); a.set_yticks([])
        if r == 0:
            a.set_title(COL_TITLES[c], fontsize=11)
        if c == 0:
            a.set_ylabel(ROW_LABELS[r], fontsize=9)

    # per-row colorbars: param scale (gridspec col 2), magnitude scale (col 7)
    cax_p = fig.add_subplot(gs[r, 2]); fig.colorbar(param_m, cax=cax_p); cax_p.tick_params(labelsize=7)
    pos = cax_p.get_position(); cax_p.set_position([pos.x0, pos.y0, 0.009, pos.height])  # thin bar at left of its wide column; rest is the gap before col-3
    cax_u = fig.add_subplot(gs[r, 7]); fig.colorbar(unc_m,   cax=cax_u); cax_u.tick_params(labelsize=7)

if SUPTITLE:
    fig.suptitle(f'Conditional-diffusion deep ensemble (M={len(members)}, '
                 f'N={NUM_SAMPLES})  —  test #{SAMPLE_IDX}, '
                 f'{"noiseless" if dbsnr is None else f"{dbsnr} dB {noise_model}"}',
                 fontsize=12, y=0.97)

if SAVE:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    noise_tag = 'noiseless' if dbsnr is None else f'{dbsnr}dB_{noise_model}'
    fname = os.path.join(OUTPUT_DIR, f'uq_idx{SAMPLE_IDX}_{noise_tag}.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f'\nsaved -> {fname}')
plt.show()
