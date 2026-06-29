"""
uq_reconstruct_all.py — batch version of uq_reconstruct.py.

Identical per-image logic (deep-ensemble aleatoric + epistemic uncertainty for
conditional-diffusion reconstruction), but loops over every test50 image and
saves each figure to OUTPUT_DIR. Runs headless (no interactive display).

See uq_reconstruct.py for the method description.
"""
import os, glob, re, time, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from slitless.forward import Imager, Source
from slitless.recon import conddiff_solver

# ── config ──────────────────────────────────────────────────────────────────────
IDXS        = range(50)       # all test50 images
NUM_SAMPLES = 10              # posterior samples per ensemble member
NOISE       = (20, 'gaussian')  # measurement noise; matches the Gaussian_20 models
SUPTITLE    = True            # show the figure suptitle
SHOW_METRICS = True           # draw per-panel [RMSE]/[MAE] metric labels under the panels

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


# ── data + ensemble ──────────────────────────────────────────────────────────────
data      = np.load(TEST50_PATH, allow_pickle=True).item()
param4dar = data['param3d'].astype(np.float32)        # (50, 3, 64, 64) physical, width Å

members = sorted(glob.glob(ENSEMBLE_GLOB))
assert members, f'no ensemble members matched {ENSEMBLE_GLOB}'
print(f'Ensemble: {len(members)} members')

dbsnr, noise_model = NOISE
noise_tag = 'noiseless' if dbsnr is None else f'{dbsnr}dB_{noise_model}'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_idx(sample_idx):
    # ── measurement (one fixed noisy realization, shared across the ensemble) ────
    Imgr = Imager(
        pixelated=True,
        spectral_orders=SPECTRAL_ORDERS,
        dispersion_scale=0.022275,
        mid_wavelength=195.119,
        dbsnr=dbsnr,
        noise_model=noise_model,
    )
    src = Source(param3d=param4dar[sample_idx].copy(), pix=False)
    Imgr.get_measurements(sources=src, no_noise=(dbsnr is None))   # sets meas3dar + srpix

    # ── sample every member on that fixed measurement ────────────────────────────
    samples_pix = []   # per member: (N, 3, 64, 64) pixel units
    for folder in members:
        ckpt = find_ckpt(folder)
        _, _, samp = conddiff_solver(
            imager=Imgr,
            run_name=os.path.basename(folder),
            model_path=ckpt,
            num_samples=NUM_SAMPLES,
            return_samples=True,
        )
        samples_pix.append(samp)
    samples_pix = np.stack(samples_pix)                            # (M, N, 3, 64, 64)

    # ── to physical units (width km/s), then decompose ───────────────────────────
    samples = Imgr.frompix(samples_pix, width_unit='km/s', array=True)

    mu_m     = samples.mean(axis=1)        # (M, 3, 64, 64)  per-member predictive mean
    v_m      = samples.var(axis=1)         # (M, 3, 64, 64)  within-member variance

    post_mean = mu_m.mean(axis=0)          # (3, 64, 64)
    ale_var   = v_m.mean(axis=0)           # aleatoric  = mean within-member variance
    epi_var   = mu_m.var(axis=0)           # epistemic  = variance of member means
    pred_var  = ale_var + epi_var          # total predictive variance

    ale_std, epi_std, pred_std = np.sqrt(ale_var), np.sqrt(epi_var), np.sqrt(pred_var)

    true = param4dar[sample_idx].copy()
    true[2] *= SPEED_OF_LIGHT / REST_WL    # width Å → km/s

    print(f'\nidx {sample_idx}  (int erg/cm²/s/sr, vel km/s, width km/s)')
    for r, name in enumerate(['int  ', 'vel  ', 'width']):
        print(f'  {name}  RMSE={rmse_ch(post_mean[r], true[r]):8.3f}   '
              f'ale={ale_std[r].mean():.3f}  epi={epi_std[r].mean():.3f}  pred={pred_std[r].mean():.3f}')

    # ── figure ────────────────────────────────────────────────────────────────────
    abs_err  = np.abs(post_mean - true)                   # (3, 64, 64)
    unc_maps = [abs_err, pred_std, ale_std, epi_std]      # order matches COL_TITLES[2:]
    fig = plt.figure(figsize=(14, 6.8))
    gs  = fig.add_gridspec(3, 8, width_ratios=[1, 1, 0.07, 1, 1, 1, 1, 0.07],
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
                    a.set_xlabel(f'[{fmt(r, rmse_ch(post_mean[r], true[r]))}]', fontsize=8)
            else:
                umap = unc_maps[c - 2][r]
                unc_m = a.imshow(umap, cmap=UNC_CMAP, vmin=0, vmax=umax)
                if SHOW_METRICS:
                    a.set_xlabel(f'[{fmt(r, umap.mean())}]', fontsize=8)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(COL_TITLES[c], fontsize=11)
            if c == 0:
                a.set_ylabel(ROW_LABELS[r], fontsize=9)

        cax_p = fig.add_subplot(gs[r, 2]); fig.colorbar(param_m, cax=cax_p); cax_p.tick_params(labelsize=7)
        cax_u = fig.add_subplot(gs[r, 7]); fig.colorbar(unc_m,   cax=cax_u); cax_u.tick_params(labelsize=7)

    if SUPTITLE:
        fig.suptitle(f'Conditional-diffusion deep ensemble (M={len(members)}, '
                     f'N={NUM_SAMPLES})  —  test #{sample_idx}, '
                     f'{"noiseless" if dbsnr is None else f"{dbsnr} dB {noise_model}"}',
                     fontsize=12, y=0.97)

    fname = os.path.join(OUTPUT_DIR, f'uq_idx{sample_idx}_{noise_tag}.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)        # free memory across the 50-image loop
    print(f'  saved -> {fname}')


# ── batch loop ────────────────────────────────────────────────────────────────────
idxs = list(IDXS)
t0 = time.time()
for n, sample_idx in enumerate(idxs):
    print('=' * 70)
    print(f'[{n+1}/{len(idxs)}] test image {sample_idx}')
    process_idx(sample_idx)
    elapsed = time.time() - t0
    eta = elapsed / (n + 1) * (len(idxs) - n - 1)
    print(f'  [elapsed {datetime.timedelta(seconds=int(elapsed))} | '
          f'eta {datetime.timedelta(seconds=int(eta))}]')

print('\nAll done in', datetime.timedelta(seconds=int(time.time() - t0)))
