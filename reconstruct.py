import json, glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization
from slitless.forward import forward_op_torch

SPEEDOFLIGHT = 299792.458
WAVELENGTH   = 195.117937907451

# ── config ────────────────────────────────────────────────────────────────────
run_folder  = './training_results/results'   # training run to load
milestone   = 10                             # model-{milestone}.pt
rec_mode    = 'all'
sample_idx  = 0                              # which dset_v6 test sample to reconstruct
numdetectors = 3
num_samples  = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = '/home/kamo/resources/slitless/data/eis_data/datasets/dset_v6/data/test'

# ── load normalization from run config ────────────────────────────────────────
config_path = f'{run_folder}/config.json'
try:
    with open(config_path) as f:
        run_config = json.load(f)
    norm_mode = run_config.get('norm_mode', 'global_logz')
    rec_mode  = run_config.get('mode', rec_mode)
except FileNotFoundError:
    norm_mode = 'global_logz'   # default for runs pre-dating config.json

normalization = make_normalization(norm_mode, rec_mode=rec_mode)

# ── load test data (physical units) ───────────────────────────────────────────
test_files = sorted(glob.glob(DATA_DIR + '/data*.npy'))
d = np.load(test_files[sample_idx], allow_pickle=True).item()

orders = [0, -1, 1, -2, 2][:numdetectors]
meas_np = np.stack([d[f'meas_{o}'] for o in orders])[None].astype(np.float32)  # (1,K,H,W)

if rec_mode == 'int':
    channels  = 1
    true_np   = d['int'][None, None].astype(np.float32)     # (1,1,H,W)
    meas_np   = meas_np[:, [0]]                              # zeroth-order only
elif rec_mode == 'vel':
    channels  = 1
    true_np   = d['vel'][None, None].astype(np.float32)
else:
    channels  = 3
    true_np   = np.stack([d['int'], d['vel'], d['width']])[None].astype(np.float32)  # (1,3,H,W)

meas = torch.tensor(meas_np).to(device)
true = torch.tensor(true_np).to(device)

# for per-sample normalization, estimate intensity scale from zeroth-order measurement
if norm_mode == 'persample_linear':
    normalization.set_infer_scale(meas[:, 0].max())

# ── forward operator (physical units → measurements) ─────────────────────────
def forward_op(x, device=None):
    if rec_mode == 'int':
        return torch.nn.Identity()(x)
    elif rec_mode == 'vel':
        int_fixed = true[:, 0].repeat(x.shape[0], 1, 1)
        wid_fixed = true[:, 2].repeat(x.shape[0], 1, 1)
        return forward_op_torch(true_intensity=int_fixed, true_doppler=x[:, 0],
                                true_linewidth=wid_fixed, device=device)
    else:
        return forward_op_torch(true_intensity=x[:, 0], true_doppler=x[:, 1],
                                true_linewidth=x[:, 2], device=device)

# ── model ─────────────────────────────────────────────────────────────────────
model = Unet(
    channels=channels,
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True
).to(device)

ckpt = torch.load(f'{run_folder}/model-{milestone}.pt', map_location=device, weights_only=True)
adapted = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
model.load_state_dict(adapted)
model.eval()

# ── diffusion + DPS ───────────────────────────────────────────────────────────
diffusion = GaussianDiffusion(
    model,
    mode=rec_mode,
    image_size=64,
    timesteps=1000,
    sampling_timesteps=1000,
    recon=True,
    measurement=meas,
    true=true,
    beta_schedule='cosine',
    clip_denoised=(-5., 5.),
    grad_scale=torch.tensor([1.]).to(device),
    forward_op=forward_op,
    device=device,
    normalization=normalization,
)

samples, norms, grad_norms, rmses = diffusion.sample(batch_size=num_samples)
samples  = samples.detach().cpu().numpy()   # (num_samples, C, H, W) — physical units
true_np  = true.detach().cpu().numpy()
rmses    = np.array(rmses).squeeze()
if len(rmses.shape) == 3:
    rmses = rmses.mean(axis=1)

# ── diagnostic plots ──────────────────────────────────────────────────────────
plt.figure(); plt.plot(norms);      plt.title('Norms');      plt.grid(); plt.show()
plt.figure(); plt.plot(grad_norms); plt.title('Grad Norms'); plt.grid(); plt.show()
plt.figure()
plt.semilogy(rmses)
plt.legend(['int', 'vel', 'width'] if rec_mode == 'all' else [rec_mode])
plt.title('RMSEs'); plt.grid(); plt.show()

# ── RMSE in physical units ────────────────────────────────────────────────────
if rec_mode == 'all':
    true_r   = true_np[0]                             # (3,H,W)
    mean_r   = samples.mean(axis=0)                   # (3,H,W)

    # convert width Å → km/s for reporting
    w_fac = SPEEDOFLIGHT / WAVELENGTH
    def rmse_ch(a, b):
        return np.sqrt(np.mean((a - b)**2, axis=(-1, -2)))

    rmse_mean = rmse_ch(true_r, mean_r)
    rmse_mean[2] *= w_fac
    rmse_all  = np.stack([rmse_ch(true_r, s) for s in samples])
    rmse_all[:, 2] *= w_fac

    print(f'RMSE (posterior mean): int={rmse_mean[0]:.1f} DN, vel={rmse_mean[1]:.2f} km/s, width={rmse_mean[2]:.2f} km/s')
    print(f'RMSE (samples mean):   int={rmse_all[:,0].mean():.1f} DN, vel={rmse_all[:,1].mean():.2f} km/s, width={rmse_all[:,2].mean():.2f} km/s')

    # ── reconstruction grid ───────────────────────────────────────────────────
    recs   = [meas[0].cpu().numpy(), true_np[0], samples[0], samples[1], samples[2], mean_r]
    titles = ['Meas', 'True', 'Sample 1', 'Sample 2', 'Sample 3', 'Posterior Mean']

    vmin0, vmax0 = true_np[0, 0].min(), true_np[0, 0].max()
    vmin1, vmax1 = true_np[0, 1].min(), true_np[0, 1].max()
    vmin2, vmax2 = true_np[0, 2].min(), true_np[0, 2].max()

    fig, ax = plt.subplots(3, 6, figsize=(15, 7))
    cmaps = ['hot', 'seismic', 'plasma']
    vmins = [vmin0, vmin1, vmin2]
    vmaxs = [vmax0, vmax1, vmax2]

    for col, (rec, title) in enumerate(zip(recs, titles)):
        for row in range(3):
            data = rec[row] if rec.shape[0] > row else rec[0]
            ax[row, col].imshow(data, cmap=cmaps[row],
                                vmin=vmins[row] if col > 0 else None,
                                vmax=vmaxs[row] if col > 0 else None)
            ax[row, col].axis('off')
            if row == 0:
                ax[row, col].set_title(title)

    cbar_ax = [fig.add_axes([0.895, 0.713, 0.01, 0.267]),
               fig.add_axes([0.895, 0.368, 0.01, 0.267]),
               fig.add_axes([0.895, 0.020, 0.01, 0.267])]
    for row in range(3):
        fig.colorbar(ax[row, 1].images[0], cax=cbar_ax[row], orientation='vertical')

    plt.tight_layout(h_pad=4, rect=[0, 0, 0.9, 1])
    plt.show()

else:
    cmap   = {'int': 'hot', 'vel': 'seismic', 'width': 'plasma'}[rec_mode]
    factor = SPEEDOFLIGHT / WAVELENGTH if rec_mode == 'width' else (34.2483 if rec_mode == 'vel' else 1)
    rmse   = np.sqrt(np.mean((true_np - samples)**2, axis=(-1, -2))) * factor
    rmse2  = np.sqrt(np.mean((true_np - samples.mean(axis=0))**2, axis=(-1, -2))) * factor

    plt.figure(); plt.imshow(true_np.squeeze(), cmap=cmap); plt.title('True'); plt.colorbar(); plt.show()
    plt.figure(); plt.imshow(samples.mean(axis=0).squeeze(), cmap=cmap); plt.title('Recon MMSE'); plt.colorbar(); plt.show()

    print(f'rmse per sample: {rmse}')
    print(f'rmse mmse: {rmse2}')
