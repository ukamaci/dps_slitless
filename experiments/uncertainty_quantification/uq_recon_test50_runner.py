"""
uq_recon_test50_runner.py — full deep-ensemble reconstruction over the test50 set.

For every test50 image, draws one fixed noisy measurement and reconstructs it with
all 8 `*_conditional_Gaussian_20` ensemble members, NUM_SAMPLES posterior samples
each (8 × 10 = 80 recons / image). All samples are saved as a single big array in
physical units (width km/s); scoring (RMSE/MAE, aleatoric/epistemic decomposition,
calibration) is done separately from this saved tensor — this script only generates
and stores the recons.

Output (in OUTPUT_DIR):
    uq_test50_recons_<tag>.npy      (50, 8, NUM_SAMPLES, 3, 64, 64) float32, physical units
    uq_test50_recons_<tag>.json     config + member order + git commit + timestamp

This is a self-contained experiment snapshot; do not edit after running.
"""
import os, glob, re, json, time, datetime, subprocess
import numpy as np
from slitless.forward import Imager, Source
from slitless.recon import conddiff_solver

# ── config ──────────────────────────────────────────────────────────────────────
IDXS               = range(50)        # all test50 images
NUM_SAMPLES        = 10               # posterior samples per ensemble member (× 8 members = 80 / image)
SAMPLING_TIMESTEPS = 250              # DDIM steps (250 matches training)
NOISE              = (20, 'gaussian')  # one fixed noisy measurement per image; matches the Gaussian_20 models

REPO_ROOT     = '/home/kamo/resources/denoising-diffusion-pytorch'   # anchor relative paths (robust to CWD)
ENSEMBLE_GLOB = f'{REPO_ROOT}/training_results/*conditional_Gaussian_20'
TEST50_PATH   = '/home/kamo/resources/slitless/data/datasets/baseline/eis_test_50_dsetv6.npy'
OUTPUT_DIR    = f'{REPO_ROOT}/experiments/uncertainty_quantification/outputs'

SPEED_OF_LIGHT  = 299792.458
REST_WL         = 195.117937907451
SPECTRAL_ORDERS = [0, -1, 1]


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


def git_commit():
    try:
        return subprocess.check_output(['git', '-C', REPO_ROOT, 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception:
        return None


# ── data + ensemble ──────────────────────────────────────────────────────────────
data      = np.load(TEST50_PATH, allow_pickle=True).item()
param4dar = data['param3d'].astype(np.float32)        # (50, 3, 64, 64) physical, width Å

members = sorted(glob.glob(ENSEMBLE_GLOB))
assert members, f'no ensemble members matched {ENSEMBLE_GLOB}'
member_names = [os.path.basename(m) for m in members]
member_ckpts = [find_ckpt(m) for m in members]
print(f'Ensemble: {len(members)} members')
for n, c in zip(member_names, member_ckpts):
    print(f'  {n}  ({c})')

dbsnr, noise_model = NOISE
noise_tag = 'noiseless' if dbsnr is None else f'{dbsnr}dB_{noise_model}'
tag       = f'8x{NUM_SAMPLES}_{SAMPLING_TIMESTEPS}ddim_{noise_tag}'

idxs = list(IDXS)
# (n_images, n_members, n_samples, 3, 64, 64) — physical units (width km/s)
recons = np.empty((len(idxs), len(members), NUM_SAMPLES, 3, 64, 64), dtype=np.float32)


def reconstruct_image(sample_idx):
    """8 members × NUM_SAMPLES posterior samples on one fixed noisy measurement.
    Returns (8, NUM_SAMPLES, 3, 64, 64) in physical units (width km/s)."""
    Imgr = Imager(
        pixelated=True,
        spectral_orders=SPECTRAL_ORDERS,
        dispersion_scale=0.022275,
        mid_wavelength=195.119,
        dbsnr=dbsnr,
        noise_model=noise_model,
    )
    src = Source(param3d=param4dar[sample_idx].copy(), pix=False)
    Imgr.get_measurements(sources=src, no_noise=(dbsnr is None))   # one fixed noisy realization, shared across members

    samples_pix = []   # per member: (N, 3, 64, 64) pixel units
    for folder in members:
        ckpt = find_ckpt(folder)
        _, _, samp = conddiff_solver(
            imager=Imgr,
            run_name=os.path.basename(folder),
            model_path=ckpt,
            num_samples=NUM_SAMPLES,
            sampling_timesteps=SAMPLING_TIMESTEPS,
            return_samples=True,
        )
        samples_pix.append(samp)
    samples_pix = np.stack(samples_pix)                            # (8, N, 3, 64, 64) pixel
    return Imgr.frompix(samples_pix, width_unit='km/s', array=True)  # (8, N, 3, 64, 64) physical


# ── batch loop ────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
t0 = time.time()
for n, sample_idx in enumerate(idxs):
    print('=' * 70)
    print(f'[{n+1}/{len(idxs)}] test image {sample_idx}')
    recons[n] = reconstruct_image(sample_idx)
    elapsed = time.time() - t0
    eta = elapsed / (n + 1) * (len(idxs) - n - 1)
    print(f'  [elapsed {datetime.timedelta(seconds=int(elapsed))} | '
          f'eta {datetime.timedelta(seconds=int(eta))}]')

# ── save ───────────────────────────────────────────────────────────────────────────
npy_path  = os.path.join(OUTPUT_DIR, f'uq_test50_recons_{tag}.npy')
json_path = os.path.join(OUTPUT_DIR, f'uq_test50_recons_{tag}.json')
np.save(npy_path, recons)

config = {
    'description': 'deep-ensemble (8 conddiff models) reconstructions over test50',
    'array_shape': list(recons.shape),
    'axes': ['image_idx (test50)', 'member', 'sample', 'channel (int/vel/width)', 'H', 'W'],
    'units': {'intensity': 'erg/cm^2/s/sr', 'velocity': 'km/s', 'width': 'km/s'},
    'idxs': idxs,
    'num_samples': NUM_SAMPLES,
    'sampling_timesteps': SAMPLING_TIMESTEPS,
    'noise': {'dbsnr': dbsnr, 'noise_model': noise_model},
    'spectral_orders': SPECTRAL_ORDERS,
    'members': member_names,
    'member_ckpts': member_ckpts,
    'ensemble_glob': ENSEMBLE_GLOB,
    'test50_path': TEST50_PATH,
    'git_commit': git_commit(),
    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    'elapsed_seconds': int(time.time() - t0),
}
with open(json_path, 'w') as f:
    json.dump(config, f, indent=2)

print('=' * 70)
print(f'saved -> {npy_path}  shape={recons.shape}')
print(f'saved -> {json_path}')
print('Total time:', datetime.timedelta(seconds=int(time.time() - t0)))
