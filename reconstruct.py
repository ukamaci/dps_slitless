import json, glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.normalization import make_normalization
from slitless.forward import forward_op_torch

SPEEDOFLIGHT     = 299792.458
WAVELENGTH       = 195.117937907451
DISPERSION_SCALE = 0.022275   # Å/pixel  (EIS: 13.5 µm pixel / (1/1.65 µm/mÅ) / 1000)
VEL_TO_PIX       = WAVELENGTH / SPEEDOFLIGHT / DISPERSION_SCALE   # km/s → pixels (~0.02922)
WIDTH_TO_PIX      = 1.0 / DISPERSION_SCALE                         # Å    → pixels (~44.89)

# ── config ────────────────────────────────────────────────────────────────────
# run_folder  = './training_results/exp_norm_logz_dset6_lr5e-6'   # training run to load
# run_folder  = './training_results/run_all_lr_1e-4_cosine_b32_logz'   # training run to load
run_folder  = './training_results/run_all_lr1e-4_cosine_b32_conditional_linear'   # training run to load
# run_folder  = './training_results/exp_norm_persample_dset6_lr5e-6'   # training run to load
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
method      = run_config.get('method', 'dps')          # 'dps' or 'conditional'
cond_orders = run_config.get('cond_orders', None)       # e.g. [0, -1, 1] for conditional

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
# forward_op_torch expects velocity and linewidth in pixel units, not km/s / Å.
def forward_op(x, device=None):
    if rec_mode == 'int':
        return torch.nn.Identity()(x)
    elif rec_mode == 'vel':
        int_fixed = true[:, 0].repeat(x.shape[0], 1, 1)
        wid_fixed = true[:, 2].repeat(x.shape[0], 1, 1)
        return forward_op_torch(true_intensity=int_fixed,
                                true_doppler=x[:, 0] * VEL_TO_PIX,
                                true_linewidth=wid_fixed * WIDTH_TO_PIX, device=device)
    else:
        return forward_op_torch(true_intensity=x[:, 0],
                                true_doppler=x[:, 1] * VEL_TO_PIX,
                                true_linewidth=x[:, 2] * WIDTH_TO_PIX, device=device)

# ── model ─────────────────────────────────────────────────────────────────────
model = Unet(
    channels=channels,
    cond_channels=len(cond_orders) if cond_orders else 0,
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True
).to(device)

ckpt = torch.load(f'{run_folder}/model-{milestone}.pt', map_location=device, weights_only=True)
adapted = {k[6:]: v for k, v in ckpt['model'].items() if k.startswith('model.')}
model.load_state_dict(adapted)
model.eval()

# ── diffusion ─────────────────────────────────────────────────────────────────
ch_labels = (['int', 'vel', 'width'] if rec_mode == 'all' else [rec_mode])
w_fac     = SPEEDOFLIGHT / WAVELENGTH

if method == 'conditional':
    cond1 = torch.stack([meas[:, cond_orders.index(o)] for o in cond_orders], dim=1) \
            if cond_orders else meas                                 # (1, K, H, W)

    diffusion = GaussianDiffusion(
        model,
        mode=rec_mode,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=250,
        beta_schedule='cosine',
        clip_denoised=(-5., 5.),
        device=device,
        normalization=normalization,
    )

    # sample one at a time; inference_mode prevents graph accumulation across steps
    samples_list = []
    for _ in range(num_samples):
        with torch.inference_mode():
            s = diffusion.sample(batch_size=1, cond=cond1)
        samples_list.append(s.cpu().numpy())
    samples = np.concatenate(samples_list, axis=0)   # (num_samples, C, H, W) — physical units

else:
    # ── DPS ───────────────────────────────────────────────────────────────────
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
        grad_scale=torch.tensor([0.5]).to(device),
        forward_op=forward_op,
        device=device,
        normalization=normalization,
    )

    samples, norms, grad_norms, rmses, ch_grad_norms = diffusion.sample(batch_size=num_samples)
    samples  = samples.detach().cpu().numpy()
    rmses    = np.array(rmses).squeeze()
    if len(rmses.shape) == 3:
        rmses = rmses.mean(axis=1)
    if rec_mode == 'all':
        rmses[:, 2] *= w_fac
    elif rec_mode == 'width':
        rmses *= w_fac

    ch_grad_norms = np.array(ch_grad_norms)   # (T, C)

    # ── DPS diagnostic plots ──────────────────────────────────────────────────
    plt.figure(); plt.plot(norms);      plt.title('Norms');      plt.grid(); plt.show()
    plt.figure(); plt.plot(grad_norms); plt.title('Grad Norms'); plt.grid(); plt.show()
    plt.figure()
    plt.semilogy(rmses)
    plt.legend(['int (erg/cm²/s/sr)', 'vel (km/s)', 'width (km/s)'] if rec_mode == 'all' else [rec_mode])
    plt.title('RMSEs'); plt.grid(); plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for c, lbl in enumerate(ch_labels):
        axes[0].semilogy(ch_grad_norms[:, c], label=lbl)
        axes[1].plot(ch_grad_norms[:, c], label=lbl)
    axes[0].set_title('Per-channel grad norms (log)'); axes[0].legend(); axes[0].grid()
    axes[1].set_title('Per-channel grad norms (linear)'); axes[1].legend(); axes[1].grid()
    plt.tight_layout(); plt.show()

    gs = diffusion.grad_scale.cpu().numpy()
    if gs.shape[0] == 1:
        gs = np.repeat(gs, len(ch_labels))
    mean_ch_gnorm = ch_grad_norms.mean(axis=0)
    print('Mean per-channel grad norm:   ' + '  '.join(f'{lbl}={v:.4f}' for lbl, v in zip(ch_labels, mean_ch_gnorm)))
    print('Effective step (gs * gnorm):  ' + '  '.join(f'{lbl}={v:.4f}' for lbl, v in zip(ch_labels, gs * mean_ch_gnorm)))

# ── RMSE in physical units ────────────────────────────────────────────────────
if rec_mode == 'all':
    true_r   = true_np[0]                             # (3,H,W)
    mean_r   = samples.mean(axis=0)                   # (3,H,W)

    def rmse_ch(a, b):
        return np.sqrt(np.mean((a - b)**2, axis=(-1, -2)))

    rmse_mean = rmse_ch(true_r, mean_r)
    rmse_mean[2] *= w_fac
    rmse_all  = np.stack([rmse_ch(true_r, s) for s in samples])
    rmse_all[:, 2] *= w_fac

    print(f'RMSE (posterior mean): int={rmse_mean[0]:.1f} erg/cm²/s/sr, vel={rmse_mean[1]:.2f} km/s, width={rmse_mean[2]:.2f} km/s')
    print(f'RMSE (samples mean):   int={rmse_all[:,0].mean():.1f} erg/cm²/s/sr, vel={rmse_all[:,1].mean():.2f} km/s, width={rmse_all[:,2].mean():.2f} km/s')

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
            cmap = 'hot' if col == 0 else cmaps[row]
            ax[row, col].imshow(data, cmap=cmap,
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
