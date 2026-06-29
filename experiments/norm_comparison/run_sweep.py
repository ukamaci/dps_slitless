import os, json, glob, time, datetime
import torch
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization
from slitless.forward import forward_op_torch

# ── constants ──────────────────────────────────────────────────────────────────
SPEEDOFLIGHT     = 299792.458
WAVELENGTH       = 195.117937907451
DISPERSION_SCALE = 0.022275
VEL_TO_PIX       = WAVELENGTH / SPEEDOFLIGHT / DISPERSION_SCALE
WIDTH_TO_PIX      = 1.0 / DISPERSION_SCALE
W_FAC             = SPEEDOFLIGHT / WAVELENGTH

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    'global_linear': {
        'run_folder': 'training_results/2026_06_23__15_52_06_all_lr_1e-4_cosine_b32_global_linear_unconditional',
        'milestone': 5,
    },
    'global_linear_pct': {
        'run_folder': 'training_results/2026_06_23__18_18_11_all_lr_1e-4_cosine_b32_global_linear_pct_unconditional',
        'milestone': 5,
    },
    'logz': {
        'run_folder': 'training_results/run_all_lr_1e-4_cosine_b32_logz',
        'milestone': 10,
    },
}

GRAD_SCALES_SWEEP = [0.1, 0.2, 0.4, 0.8]  # applied equally to all channels
LOGZ_GRAD_SCALE      = 0.5                  # fixed baseline from ICASSP paper
NUM_SAMPLES          = 10
SAMPLING_TIMESTEPS   = 1000                 # full DDPM
GRAD_NORM_MODE       = 'int_slope'          # for global_linear* models
LOGZ_GRAD_NORM_MODE  = 'meas_rms'          # logz uses meas_rms (matches dps_solver default)
ORDERS             = [0, -1, 1]
NUM_VAL            = 25

DATA_DIR_VAL = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data/val'
TEST50_PATH  = '/home/kamo/resources/slitless/data/datasets/baseline/eis_test_50_dsetv6.npy'
OUTPUT_DIR   = 'experiments/norm_comparison/outputs'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ── forward operator ───────────────────────────────────────────────────────────
def forward_op(x, device=None):
    return forward_op_torch(
        true_intensity=x[:, 0],
        true_doppler=x[:, 1] * VEL_TO_PIX,
        true_linewidth=x[:, 2] * WIDTH_TO_PIX,
        device=device,
    )


# ── create/load val_25 ─────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

val25_path = os.path.join(OUTPUT_DIR, 'val_25.npy')
if not os.path.exists(val25_path):
    print('Creating val_25 dataset (first 25 sorted val files)...')
    files = sorted(glob.glob(os.path.join(DATA_DIR_VAL, 'data*.npy')))[:NUM_VAL]
    assert len(files) == NUM_VAL, f'Expected {NUM_VAL} val files, got {len(files)}'
    meas_list, param_list = [], []
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        meas_list.append(np.stack([d[f'meas_{o}'] for o in ORDERS]))
        param_list.append(np.stack([d['int'], d['vel'], d['width']]))
    np.save(val25_path, {
        'meas':   np.array(meas_list,  dtype=np.float32),   # (25,3,64,64)
        'param3d': np.array(param_list, dtype=np.float32),  # (25,3,64,64)
    })
    print(f'  saved -> {val25_path}')
else:
    print(f'val_25 already exists: {val25_path}')

val25      = np.load(val25_path, allow_pickle=True).item()
val25_meas = torch.tensor(val25['meas']).to(device)     # (25,3,64,64)
val25_true = torch.tensor(val25['param3d']).to(device)  # (25,3,64,64)

# ── load test_50 ───────────────────────────────────────────────────────────────
print('Loading test_50...')
test50       = np.load(TEST50_PATH, allow_pickle=True).item()
test50_meas  = torch.tensor(test50['meas'][:, :3].astype(np.float32)).to(device)  # (50,3,64,64)
test50_true  = torch.tensor(test50['param3d'].astype(np.float32)).to(device)      # (50,3,64,64)
N_TEST       = len(test50_true)


# ── helpers ────────────────────────────────────────────────────────────────────
def load_weights(model, run_folder, milestone):
    ckpt  = torch.load(f'{run_folder}/model-{milestone}.pt', map_location=device, weights_only=True)
    state = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
    model.load_state_dict(state)
    model.eval()


def run_dps(diffusion, meas_i):
    """meas_i: (1,3,64,64) on device. Returns posterior mean (3,64,64) numpy, physical units."""
    diffusion.measurement = meas_i
    samples, *_ = diffusion.sample(batch_size=NUM_SAMPLES)
    return samples.detach().cpu().numpy().mean(axis=0)  # (3,64,64)


def rmse_phy(mean_recon, true_np):
    """Both (3,64,64) numpy in physical units; width in Å → converted to km/s in output."""
    r = np.sqrt(np.mean((mean_recon - true_np) ** 2, axis=(-1, -2)))  # (3,)
    r[2] *= W_FAC
    return r


def make_diffusion(model, normalization, clip_denoised, dummy_meas, gs, gnm=GRAD_NORM_MODE):
    return GaussianDiffusion(
        model,
        mode='all',
        image_size=64,
        timesteps=1000,
        sampling_timesteps=SAMPLING_TIMESTEPS,
        recon=True,
        measurement=dummy_meas,
        true=None,                  # skip per-step RMSE tracking
        beta_schedule='cosine',
        clip_denoised=clip_denoised,
        grad_scale=torch.tensor([gs, gs, gs]).to(device),
        grad_norm_mode=gnm,
        forward_op=forward_op,
        device=device,
        normalization=normalization,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — val-25 grad_scale sweep (global_linear + global_linear_pct)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print(' PHASE 1: val-25 grad_scale sweep')
print('='*70)

sweep_results = {}
global_start = time.time()

for model_name in ['global_linear', 'global_linear_pct']:
    cfg = MODEL_CONFIGS[model_name]
    print(f'\n── {model_name}  (milestone {cfg["milestone"]}) ──')

    with open(f'{cfg["run_folder"]}/config.json') as f:
        run_cfg = json.load(f)
    norm_mode     = run_cfg['norm_mode']
    normalization = make_normalization(norm_mode, rec_mode='all')
    clip_denoised = tuple(run_cfg.get('clip_denoised', normalization.clip_denoised))

    model = Unet(channels=3, dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(device)
    load_weights(model, cfg['run_folder'], cfg['milestone'])

    sweep_results[model_name] = {}

    for gs in GRAD_SCALES_SWEEP:
        print(f'\n  grad_scale = {gs}')
        diffusion = make_diffusion(model, normalization, clip_denoised, val25_meas[[0]], gs)
        rmses = []
        t0 = time.time()

        for i in range(NUM_VAL):
            mean_recon = run_dps(diffusion, val25_meas[[i]])
            rmses.append(rmse_phy(mean_recon, val25_true[i].cpu().numpy()))
            torch.cuda.empty_cache()
            if (i + 1) % 5 == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (i + 1) * (NUM_VAL - i - 1)
                print(f'    [{i+1:2d}/{NUM_VAL}] vel={rmses[-1][1]:.3f} km/s  '
                      f'[elapsed {datetime.timedelta(seconds=int(elapsed))}'
                      f' | eta {datetime.timedelta(seconds=int(eta))}]')

        rmses = np.array(rmses)  # (25,3)
        sweep_results[model_name][gs] = rmses
        mu = rmses.mean(axis=0)
        print(f'    mean RMSE: int={mu[0]:.1f}  vel={mu[1]:.3f}  width={mu[2]:.3f} km/s')
        del diffusion

    del model
    torch.cuda.empty_cache()

# best grad_scale per model = lowest mean vel RMSE on val-25
best_gs = {}
for model_name in ['global_linear', 'global_linear_pct']:
    vel_means = {gs: sweep_results[model_name][gs][:, 1].mean() for gs in GRAD_SCALES_SWEEP}
    best_gs[model_name] = min(vel_means, key=vel_means.get)
    print(f'\nBest grad_scale for {model_name}: {best_gs[model_name]}'
          f'  (val vel RMSE = {vel_means[best_gs[model_name]]:.3f} km/s)')

np.save(os.path.join(OUTPUT_DIR, 'val25_sweep.npy'), {
    'config': {
        'models':            {k: MODEL_CONFIGS[k] for k in ['global_linear', 'global_linear_pct']},
        'grad_scales':       GRAD_SCALES_SWEEP,
        'num_samples':       NUM_SAMPLES,
        'sampling_timesteps': SAMPLING_TIMESTEPS,
        'grad_norm_mode':    GRAD_NORM_MODE,
        'best_grad_scales':  best_gs,
    },
    'results': sweep_results,
})
print(f'\nSaved val25_sweep -> {OUTPUT_DIR}/val25_sweep.npy')
phase1_elapsed = time.time() - global_start
print(f'Phase 1 done in {datetime.timedelta(seconds=int(phase1_elapsed))}')


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — test-50: best of each model + logz baseline
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print(' PHASE 2: test-50 — best configs + logz baseline')
print('='*70)

test50_runs = {
    'global_linear':     (MODEL_CONFIGS['global_linear'],     best_gs['global_linear']),
    'global_linear_pct': (MODEL_CONFIGS['global_linear_pct'], best_gs['global_linear_pct']),
    'logz':              (MODEL_CONFIGS['logz'],              LOGZ_GRAD_SCALE),
}

test50_results = {}
truths_np = test50_true.cpu().numpy()  # (50,3,64,64)

for method_name, (cfg, gs) in test50_runs.items():
    print(f'\n── {method_name}  gs={gs}  milestone={cfg["milestone"]} ──')

    with open(f'{cfg["run_folder"]}/config.json') as f:
        run_cfg = json.load(f)
    norm_mode     = run_cfg['norm_mode']
    normalization = make_normalization(norm_mode, rec_mode='all')
    clip_denoised = tuple(run_cfg.get('clip_denoised', normalization.clip_denoised))

    gnm = LOGZ_GRAD_NORM_MODE if method_name == 'logz' else GRAD_NORM_MODE
    model = Unet(channels=3, dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(device)
    load_weights(model, cfg['run_folder'], cfg['milestone'])
    diffusion = make_diffusion(model, normalization, clip_denoised, test50_meas[[0]], gs, gnm=gnm)

    rmses, recons = [], []
    t0 = time.time()

    for i in range(N_TEST):
        mean_recon = run_dps(diffusion, test50_meas[[i]])
        rmses.append(rmse_phy(mean_recon, truths_np[i]))
        recons.append(mean_recon)
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (N_TEST - i - 1)
            print(f'  [{i+1:2d}/{N_TEST}] vel={rmses[-1][1]:.3f} km/s  '
                  f'[elapsed {datetime.timedelta(seconds=int(elapsed))}'
                  f' | eta {datetime.timedelta(seconds=int(eta))}]')

    rmses = np.array(rmses)  # (50,3) — width already in km/s
    mu    = rmses.mean(axis=0)
    print(f'  mean RMSE: int={mu[0]:.1f}  vel={mu[1]:.3f}  width={mu[2]:.3f} km/s')

    test50_results[method_name] = {
        'grad_scale':  gs,
        'rmses':       rmses,              # (50,3) physical (width in km/s)
        'recons_mean': np.array(recons),   # (50,3,64,64) — width in Å from unnormalize
    }

    del model, diffusion
    torch.cuda.empty_cache()

np.save(os.path.join(OUTPUT_DIR, 'test50_results.npy'), {
    'config': {
        'runs': {k: {'run_folder': v[0]['run_folder'], 'milestone': v[0]['milestone'],
                     'grad_scale': v[1]} for k, v in test50_runs.items()},
        'num_samples':        NUM_SAMPLES,
        'sampling_timesteps': SAMPLING_TIMESTEPS,
        'grad_norm_mode':     GRAD_NORM_MODE,
    },
    'results': test50_results,
    'truths':  truths_np,   # (50,3,64,64) — int(erg), vel(km/s), width(Å)
})
print(f'\nSaved test50_results -> {OUTPUT_DIR}/test50_results.npy')

total_elapsed = time.time() - global_start
print(f'\nAll done in {datetime.timedelta(seconds=int(total_elapsed))}.')
