"""
uq_ood_reconstruct.py — OOD-detection variant of uq_reconstruct.py.

A small UIUC "I" phantom is stamped into an in-distribution EIS test image
(overwriting intensity / velocity / line-width on the "I" pixels with chosen
out-of-distribution values), the measurement is re-simulated, and the same
deep-ensemble conditional-diffusion reconstruction + uncertainty decomposition
is run. The injected "I" is locally out-of-distribution, so the epistemic map
(between-model disagreement) is expected to light up over it while the rest of
the scene stays low — i.e. epistemic uncertainty as an OOD detector.

Figure is identical to uq_reconstruct.py (3×6):
    True | Posterior Mean | Posterior Mean Error | Total Unc. | Aleatoric | Epistemic
with rows = channels (int / vel / width). "True" here is the OOD-injected map.
See uq_reconstruct.py for the method description and bracket conventions.
"""
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import uniform_filter
from PIL import Image
from slitless.forward import Imager, Source
from slitless.recon import conddiff_solver
from slitless.plotting import uiuc_i, sincosgrid

# ── config ──────────────────────────────────────────────────────────────────────
SAMPLE_IDX  = 20              # in-distribution test50 image to inject into
NUM_SAMPLES = 10              # posterior samples per ensemble member
SAMPLING_TIMESTEPS = 250      # DDIM sampling steps (250 matches training)
NOISE       = (20, 'gaussian')  # measurement noise; matches the Gaussian_20 models
# NOISE       = (None, None)  # measurement noise; matches the Gaussian_20 models
SAVE        = True
SUPTITLE    = False
SHOW_MEAS   = False           # prepend a Measurements column (3 diffraction orders); shares the intensity colorbar
SHOW_METRICS = False           # draw per-panel RMSE/MAE metric labels inside the panels
COMPARISON  = True            # also reconstruct the clean (no-I) image and report its metrics, to isolate the OOD-patch effect
OOD_MAP     = True            # extra figure: epistemic/aleatoric ratio map ("OOD map")
OOD_WINDOW  = 5               # box-filter window (px) for the smoothed ratio; 1 = pixelwise (prefer odd sizes)

# ── injected "I" phantom ─────────────────────────────────────────────────────────
I_HEIGHT    = 10              # height of the inserted "I" in pixels (aspect preserved)
I_POS       = (6, 8)         # (row, col) of the I-patch top-left corner
# I_HEIGHT    = 47              # height of the inserted "I" in pixels (aspect preserved)
# I_POS       = (8, 17)         # (row, col) of the I-patch top-left corner
I_INTENSITY = 40.0         # injected intensity   [erg/cm²/s/sr]
I_VELOCITY  = 40.0           # injected Doppler vel  [km/s]
I_WIDTH     = 50.0            # injected line width   [km/s]
I_MODULATE  = False         # texture the I with a 2D cos·sin grid (à la uiuc_im)
I_MOD_TX    = 9              # cosine periods along x
I_MOD_TY    = 9              # cosine periods along y
I_MOD_AMP   = 0.15           # modulation depth (fraction of the base value)

REPO_ROOT     = '/home/kamo/resources/denoising-diffusion-pytorch'   # anchor relative paths (robust to CWD when run from a notebook)
ENSEMBLE_GLOB = f'{REPO_ROOT}/training_results/*conditional_Gaussian_20'
TEST50_PATH   = '/home/kamo/resources/slitless/data/datasets/baseline/eis_test_50_dsetv6.npy'
OUTPUT_DIR    = f'{REPO_ROOT}/experiments/uncertainty_quantification/outputs'

SPEED_OF_LIGHT  = 299792.458
REST_WL         = 195.117937907451
SPECTRAL_ORDERS = [0, -1, 1]
W_FAC           = SPEED_OF_LIGHT / REST_WL     # km/s per Å

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


def mae_ch(a, b):
    return np.mean(np.abs(a - b))


def ood_map(ale_std, epi_std, window):
    """Epistemic/aleatoric ratio per channel — an "OOD map". With window>1 each
    std map is first box-averaged (uniform_filter) over a window×window neighborhood
    and *then* divided, i.e. a ratio of local means (matches the region-averaged
    ratio). window=1 gives the pixelwise ratio. Returns (3, H, W)."""
    if window > 1:
        ale_std = np.stack([uniform_filter(a, size=window, mode='nearest') for a in ale_std])
        epi_std = np.stack([uniform_filter(e, size=window, mode='nearest') for e in epi_std])
    return epi_std / (ale_std + 1e-8)


def panel_label(ax, text):
    """Draw a metric label inside the panel (mid-bottom), large and readable on
    any colormap via white text with a black outline."""
    t = ax.text(0.5, 0.03, text, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=16, fontweight='bold', color='white')
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground='black')])


def small_i_mask(height):
    """Crop uiuc_i() to its bounding box and resize to `height` px (aspect kept).
    Returns a binary {0,1} mask of the I silhouette."""
    m = uiuc_i()                                   # (64,64), {0,0.502,1.0}
    ys, xs = np.where(m > 0)
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h0, w0 = crop.shape
    w_new = max(1, round(w0 * height / h0))
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize((w_new, height), Image.NEAREST)
    return (np.array(img) / 255.0 > 0.25).astype(np.float32)


def inject_i(param3d, mask, pos, vals, mod=None):
    """Stamp the I `mask` into param3d (3,H,W) at top-left `pos`, setting the
    masked pixels of each channel to `vals` = [int, vel(km/s), width(Å)].
    If `mod` (64×64) is given, the values are multiplied by (1 + mod) at each
    pixel, texturing the I with the 2D cos·sin grid. Clips to bounds; returns a copy."""
    out = param3d.copy()
    r0, c0 = pos
    h, w = mask.shape
    r1, c1 = min(r0 + h, out.shape[1]), min(c0 + w, out.shape[2])
    region = mask[:r1 - r0, :c1 - c0] > 0.5
    factor = np.ones((r1 - r0, c1 - c0), np.float32) if mod is None else (1.0 + mod[r0:r1, c0:c1])
    for ch in range(3):
        out[ch, r0:r1, c0:c1][region] = vals[ch] * factor[region]
    return out


# ── data + ensemble ──────────────────────────────────────────────────────────────
data      = np.load(TEST50_PATH, allow_pickle=True).item()
param4dar = data['param3d'].astype(np.float32)        # (50, 3, 64, 64) physical, width Å

members = sorted(glob.glob(ENSEMBLE_GLOB))
assert members, f'no ensemble members matched {ENSEMBLE_GLOB}'
print(f'Ensemble: {len(members)} members')

# ── build the OOD-injected ground truth (width handled in Å internally) ──────────
mask = small_i_mask(I_HEIGHT)
mod  = (I_MOD_AMP * sincosgrid(64, I_MOD_TX, I_MOD_TY)).astype(np.float32) if I_MODULATE else None
print(f'Injected "I": {mask.shape[0]}×{mask.shape[1]} px at top-left {I_POS}  '
      f'(int={I_INTENSITY}, vel={I_VELOCITY} km/s, width={I_WIDTH} km/s)'
      f'{f"  +cos·sin texture tx={I_MOD_TX} ty={I_MOD_TY} amp={I_MOD_AMP}" if I_MODULATE else ""}')
param_ood = inject_i(
    param4dar[SAMPLE_IDX],
    mask,
    I_POS,
    vals=[I_INTENSITY, I_VELOCITY, I_WIDTH / W_FAC],   # width km/s → Å
    mod=mod,
)

dbsnr, noise_model = NOISE
noise_tag = 'noiseless' if dbsnr is None else f'{dbsnr}dB_{noise_model}'


def run_recon(param_phys):
    """Reconstruct one ground-truth param map (3,64,64; width in Å) with the full
    ensemble on one fixed noisy measurement. Returns physical-unit (width km/s)
    posterior-mean and uncertainty maps, the km/s ground truth, and the measurement."""
    Imgr = Imager(
        pixelated=True,
        spectral_orders=SPECTRAL_ORDERS,
        dispersion_scale=0.022275,
        mid_wavelength=195.119,
        dbsnr=dbsnr,
        noise_model=noise_model,
    )
    src = Source(param3d=param_phys.copy(), pix=False)
    Imgr.get_measurements(sources=src, no_noise=(dbsnr is None))   # sets meas3dar + srpix

    samples_pix = []   # per member: (N, 3, 64, 64) pixel units
    for i, folder in enumerate(members):
        ckpt = find_ckpt(folder)
        print(f'  [{i+1}/{len(members)}] {os.path.basename(folder)}  ({ckpt})')
        _, _, samp = conddiff_solver(
            imager=Imgr,
            run_name=os.path.basename(folder),
            model_path=ckpt,
            num_samples=NUM_SAMPLES,
            sampling_timesteps=SAMPLING_TIMESTEPS,
            return_samples=True,
        )
        samples_pix.append(samp)
    samples_pix = np.stack(samples_pix)                           # (M, N, 3, 64, 64)

    samples = Imgr.frompix(samples_pix, width_unit='km/s', array=True)  # to physical (width km/s)
    mu_m = samples.mean(axis=1)         # (M, 3, 64, 64)  per-member predictive mean
    v_m  = samples.var(axis=1)          # (M, 3, 64, 64)  within-member variance

    post_mean = mu_m.mean(axis=0)       # (3, 64, 64)
    ale_var   = v_m.mean(axis=0)        # aleatoric = mean within-member variance
    epi_var   = mu_m.var(axis=0)        # epistemic = variance of member means
    pred_var  = ale_var + epi_var       # total predictive variance

    true = param_phys.copy()
    true[2] *= W_FAC                    # width Å → km/s
    meas = Imgr.meas3dar
    meas = meas.detach().cpu().numpy() if hasattr(meas, 'detach') else np.asarray(meas)

    return dict(post_mean=post_mean, true=true, meas=meas,
                ale_std=np.sqrt(ale_var), epi_std=np.sqrt(epi_var), pred_std=np.sqrt(pred_var))


# ── injected "I" neighborhood mask (whole-image vs. region split) ────────────────
# Use a rectangle 1.5× the I bounding box, centered on it: the injection perturbs
# the reconstruction beyond the exact silhouette, so neighboring pixels count too.
H, W = mask.shape
r0, c0 = I_POS
HH, WW = param4dar.shape[2], param4dar.shape[3]
cr, cc = r0 + H / 2.0, c0 + W / 2.0
H2, W2 = round(1.5 * H), round(1.5 * W)
rr0, rr1 = max(0, round(cr - H2 / 2.0)), min(HH, round(cr + H2 / 2.0))
cc0, cc1 = max(0, round(cc - W2 / 2.0)), min(WW, round(cc + W2 / 2.0))
imask = np.zeros((HH, WW), dtype=bool)
imask[rr0:rr1, cc0:cc1] = True


def summarize(tag, res):
    """Per-channel RMSE/MAE + mean uncertainty for one recon, on the whole image
    and inside the injected-"I" region. Returns a list of text lines."""
    lines = [f'{tag}  (int erg/cm²/s/sr, vel km/s, width km/s)']
    for region_name, m in (('whole image', None), ('injected "I" region (1.5x bbox)', imask)):
        lines.append(f'  {region_name}:')
        for r, name in enumerate(['int  ', 'vel  ', 'width']):
            pm, tr = res['post_mean'][r], res['true'][r]
            als, eps, prs = res['ale_std'][r], res['epi_std'][r], res['pred_std'][r]
            if m is not None:
                pm, tr, als, eps, prs = pm[m], tr[m], als[m], eps[m], prs[m]
            lines.append(
                f'    {name}  RMSE={rmse_ch(pm, tr):8.3f}  MAE={mae_ch(pm, tr):8.3f}   '
                f'ale={als.mean():.3f}  epi={eps.mean():.3f}  pred={prs.mean():.3f}')
    return lines


# ── reconstruct: OOD-injected image (always), clean image (if COMPARISON) ────────
print('Reconstructing OOD-injected image:')
res_ood = run_recon(param_ood)
res_clean = None
if COMPARISON:
    print('Reconstructing clean (no-I) image:')
    res_clean = run_recon(param4dar[SAMPLE_IDX])

# the figure shows the OOD reconstruction
post_mean, true, meas = res_ood['post_mean'], res_ood['true'], res_ood['meas']
ale_std, epi_std, pred_std = res_ood['ale_std'], res_ood['epi_std'], res_ood['pred_std']

# ── console summary (whole image + the injected "I" neighborhood) ────────────────
summary_lines = summarize(f'OOD-injected   idx {SAMPLE_IDX}', res_ood)
if COMPARISON:
    summary_lines.append('')
    summary_lines += summarize(f'clean (no I)   idx {SAMPLE_IDX}', res_clean)
summary = '\n'.join(summary_lines)
print('\n' + summary)

# ── figure (identical layout to uq_reconstruct.py) ──────────────────────────────
abs_err  = np.abs(post_mean - true)                   # (3, 64, 64)
unc_maps = [abs_err, pred_std, ale_std, epi_std]      # order matches COL_TITLES[2:]

# optional Measurements column: one diffraction order per row, all intensity-like (OOD recon)
col_titles = (['Measurements'] if SHOW_MEAS else []) + COL_TITLES

if SHOW_MEAS:
    # extend the row-0 intensity colorbar so it also covers the measurement frames
    int_lo = min(float(true[0].min()), float(meas.min()))
    int_hi = max(float(true[0].max()), float(meas.max()))
    width_ratios = [1, 1, 1, 0.45, 1, 1, 1, 1, 0.07]   # wide param-cb col → room for ticks before col-3
    IMG_GC = [0, 1, 2, 4, 5, 6, 7]   # Meas, True, PostMean, Err, Total, Ale, Epi
    PARAM_CB_GC, UNC_CB_GC = 3, 8
else:
    width_ratios = [1, 1, 0.2, 1, 1, 1, 1, 0.07]      # wide param-cb col → room for ticks before col-3
    IMG_GC = [0, 1, 3, 4, 5, 6]      # True, PostMean, Err, Total, Ale, Epi
    PARAM_CB_GC, UNC_CB_GC = 2, 7

fig = plt.figure(figsize=(16 if SHOW_MEAS else 14, 6.8))
gs  = fig.add_gridspec(3, len(width_ratios), width_ratios=width_ratios,
                       left=0.06, right=0.96, top=0.90, bottom=0.05,
                       hspace=0.10, wspace=0.08)

for r in range(3):
    vmin, vmax = true[r].min(), true[r].max()
    if r == 0 and SHOW_MEAS:                          # intensity row shares the meas scale
        vmin, vmax = int_lo, int_hi
    umax = pred_std[r].max()                          # shared magnitude scale per row
    param_m = unc_m = None
    for c, gc in enumerate(IMG_GC):
        a = fig.add_subplot(gs[r, gc])
        title = col_titles[c]
        if title == 'Measurements':
            a.imshow(meas[r], cmap=CMAPS[0], vmin=int_lo, vmax=int_hi)
            a.set_xlabel(f'order {SPECTRAL_ORDERS[r]}', fontsize=8)
        elif title == 'True':
            param_m = a.imshow(true[r], cmap=CMAPS[r], vmin=vmin, vmax=vmax)
        elif title == 'Posterior Mean':
            a.imshow(post_mean[r], cmap=CMAPS[r], vmin=vmin, vmax=vmax)
            if SHOW_METRICS:
                panel_label(a, fmt(r, rmse_ch(post_mean[r], true[r])))
        else:
            umap = unc_maps[COL_TITLES.index(title) - 2][r]
            unc_m = a.imshow(umap, cmap=UNC_CMAP, vmin=0, vmax=umax)
            if SHOW_METRICS:
                panel_label(a, fmt(r, umap.mean()))
        a.add_patch(plt.Rectangle((cc0 - 0.5, rr0 - 0.5), cc1 - cc0, rr1 - rr0,
                                  fill=False, edgecolor='lime', lw=0.8))  # injected "I" region (1.5x bbox)
        a.set_xticks([]); a.set_yticks([])
        if r == 0:
            a.set_title(title, fontsize=11)
        if c == 0:
            a.set_ylabel(ROW_LABELS[r], fontsize=9)

    cax_p = fig.add_subplot(gs[r, PARAM_CB_GC]); fig.colorbar(param_m, cax=cax_p); cax_p.tick_params(labelsize=7)
    pos = cax_p.get_position(); cax_p.set_position([pos.x0, pos.y0, 0.009, pos.height])  # thin bar at left of its wide column; rest is the gap before col-3
    cax_u = fig.add_subplot(gs[r, UNC_CB_GC]); fig.colorbar(unc_m,   cax=cax_u); cax_u.tick_params(labelsize=7)

if SUPTITLE:
    fig.suptitle(f'Conditional-diffusion deep ensemble (M={len(members)}, N={NUM_SAMPLES}) '
                 f'— OOD "I" injected into test #{SAMPLE_IDX}, '
                 f'{"noiseless" if dbsnr is None else f"{dbsnr} dB {noise_model}"}',
                 fontsize=12, y=0.97)

if SAVE:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = os.path.join(OUTPUT_DIR, f'uq_ood_idx{SAMPLE_IDX}_I{I_HEIGHT}_{noise_tag}.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    txt_fname = os.path.splitext(fname)[0] + '.txt'
    with open(txt_fname, 'w') as f:
        f.write(summary + '\n')
    print(f'\nsaved -> {fname}')
    print(f'saved -> {txt_fname}')

# ── optional OOD map: epistemic/aleatoric ratio (OOD recon, clean recon below) ───
if OOD_MAP:
    rows = [('OOD-injected', res_ood)] + ([('clean (no I)', res_clean)] if COMPARISON else [])
    omaps = [ood_map(res['ale_std'], res['epi_std'], OOD_WINDOW) for _, res in rows]
    vmaxes = [np.percentile(omaps[0][r], 99) for r in range(3)]   # scale by OOD map so rows are comparable
    figo, axo = plt.subplots(len(rows), 3, figsize=(11, 3.6 * len(rows)), squeeze=False)
    for ri, (label, _) in enumerate(rows):
        for r in range(3):
            a = axo[ri][r]
            im = a.imshow(omaps[ri][r], cmap='inferno', vmin=0, vmax=vmaxes[r])
            a.add_patch(plt.Rectangle((cc0 - 0.5, rr0 - 0.5), cc1 - cc0, rr1 - rr0,
                                      fill=False, edgecolor='lime', lw=0.8))  # injected "I" region (1.5x bbox)
            a.set_xticks([]); a.set_yticks([])
            if ri == 0:
                a.set_title(ROW_LABELS[r].replace('\n', ' '), fontsize=10)
            if r == 0:
                a.set_ylabel(label, fontsize=10)
            figo.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    figo.suptitle(f'OOD map: epistemic / aleatoric  (window={OOD_WINDOW})', fontsize=12)
    figo.tight_layout()
    if SAVE:
        oname = os.path.join(OUTPUT_DIR,
                             f'uq_ood_idx{SAMPLE_IDX}_I{I_HEIGHT}_{noise_tag}_oodmap_w{OOD_WINDOW}.png')
        figo.savefig(oname, dpi=200, bbox_inches='tight')
        print(f'saved -> {oname}')

plt.show()