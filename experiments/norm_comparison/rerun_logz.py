"""Re-run logz on test-50 with the correct grad_norm_mode='meas_rms'.

The original run_sweep.py used 'int_slope' for logz, which gave
meas_scale=0.682 — making the DPS step ~5800x larger than intended.
The reference dps_solver uses 'meas_rms' (the GaussianDiffusion default).
"""
import os, json, time, datetime
import torch
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization
from slitless.forward import forward_op_torch

SPEEDOFLIGHT     = 299792.458
WAVELENGTH       = 195.117937907451
DISPERSION_SCALE = 0.022275
VEL_TO_PIX       = WAVELENGTH / SPEEDOFLIGHT / DISPERSION_SCALE
WIDTH_TO_PIX      = 1.0 / DISPERSION_SCALE
W_FAC             = SPEEDOFLIGHT / WAVELENGTH

OUTPUT_DIR   = 'experiments/norm_comparison/outputs'
TEST50_PATH  = '/home/kamo/resources/slitless/data/datasets/baseline/eis_test_50_dsetv6.npy'

LOGZ_RUN     = 'training_results/run_all_lr_1e-4_cosine_b32_logz'
LOGZ_MS      = 10
LOGZ_GS      = 0.5
LOGZ_GNM     = 'meas_rms'
NUM_SAMPLES  = 10
ORDERS       = [0, -1, 1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

def forward_op(x, device=None):
    return forward_op_torch(
        true_intensity=x[:, 0],
        true_doppler=x[:, 1] * VEL_TO_PIX,
        true_linewidth=x[:, 2] * WIDTH_TO_PIX,
        device=device,
    )

def rmse_phy(mean_recon, true_np):
    r = np.sqrt(np.mean((mean_recon - true_np) ** 2, axis=(-1, -2)))
    r[2] *= W_FAC
    return r

# ── load test_50 ──────────────────────────────────────────────────────────────
test50      = np.load(TEST50_PATH, allow_pickle=True).item()
test50_meas = torch.tensor(test50['meas'][:, :3].astype(np.float32)).to(device)
truths_np   = test50['param3d'].astype(np.float32)
N_TEST      = len(truths_np)

# ── load model ────────────────────────────────────────────────────────────────
with open(f'{LOGZ_RUN}/config.json') as f:
    run_cfg = json.load(f)
norm_mode     = run_cfg['norm_mode']
normalization = make_normalization(norm_mode, rec_mode='all')
clip_denoised = tuple(run_cfg.get('clip_denoised', normalization.clip_denoised))

model = Unet(channels=3, dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(device)
ckpt  = torch.load(f'{LOGZ_RUN}/model-{LOGZ_MS}.pt', map_location=device, weights_only=True)
state = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
model.load_state_dict(state)
model.eval()

# ── build diffusion ───────────────────────────────────────────────────────────
diffusion = GaussianDiffusion(
    model,
    mode='all',
    image_size=64,
    timesteps=1000,
    sampling_timesteps=1000,
    recon=True,
    measurement=test50_meas[[0]],
    true=None,
    beta_schedule='cosine',
    clip_denoised=clip_denoised,
    grad_scale=torch.tensor([LOGZ_GS, LOGZ_GS, LOGZ_GS]).to(device),
    grad_norm_mode=LOGZ_GNM,
    forward_op=forward_op,
    device=device,
    normalization=normalization,
)
print(f'meas_scale (meas_rms on first image): {diffusion.meas_scale:.3f}')

# ── run ───────────────────────────────────────────────────────────────────────
rmses, recons = [], []
t0 = time.time()
for i in range(N_TEST):
    diffusion.measurement = test50_meas[[i]]
    if LOGZ_GNM == 'meas_rms':
        diffusion.meas_scale = test50_meas[[i]].pow(2).mean().sqrt().item()
    samples, *_ = diffusion.sample(batch_size=NUM_SAMPLES)
    mean_recon = samples.detach().cpu().numpy().mean(axis=0)
    rmses.append(rmse_phy(mean_recon, truths_np[i]))
    recons.append(mean_recon)
    torch.cuda.empty_cache()
    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (N_TEST - i - 1)
        print(f'  [{i+1:2d}/{N_TEST}] vel={rmses[-1][1]:.3f} km/s  '
              f'[elapsed {datetime.timedelta(seconds=int(elapsed))}'
              f' | eta {datetime.timedelta(seconds=int(eta))}]')

rmses = np.array(rmses)
mu    = rmses.mean(axis=0)
print(f'\nmean RMSE: int={mu[0]:.1f}  vel={mu[1]:.3f}  width={mu[2]:.3f} km/s')
print(f'[int infs: {np.isinf(rmses[:,0]).sum()}, nans: {np.isnan(rmses[:,0]).sum()}]')

# ── patch test50_results.npy ──────────────────────────────────────────────────
existing = np.load(os.path.join(OUTPUT_DIR, 'test50_results.npy'), allow_pickle=True).item()
existing['results']['logz'] = {
    'grad_scale':   LOGZ_GS,
    'grad_norm_mode': LOGZ_GNM,
    'rmses':        rmses,
    'recons_mean':  np.array(recons),
}
existing['config']['runs']['logz']['grad_norm_mode'] = LOGZ_GNM
np.save(os.path.join(OUTPUT_DIR, 'test50_results.npy'), existing)
print(f'\nPatched test50_results.npy with corrected logz results.')
print(f'Re-run analyze.py to regenerate tables and scatter plots.')
