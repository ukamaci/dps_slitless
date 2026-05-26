import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.plotting import plotgrid
import matplotlib.pyplot as plt
from slitless.data_loader import BasicDataset
from torch.utils.data import DataLoader
from slitless.forward import Source
import numpy as np
from pathlib import Path

# %% Config
mode = 'all'
run_folder = Path('./training_results/run_all_lr5e-6_cosine_b32')
modellist = [10]
# modellist = [10,20,30,40,50]
# modellist = np.concatenate((np.array([1]), np.linspace(5,70,14).astype(int)))
num_samples = 5
save = False   # save PNGs to run_folder/sampled_images/ instead of displaying

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels = 3 if mode == 'all' else 1

if save:
    sampled_images_folder = run_folder / 'sampled_images'
    sampled_images_folder.mkdir(exist_ok=True)

# %% DDPM Sampling

model = Unet(
    dim = 64,
    channels = channels,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
).to(device)

for modelnum in modellist:
    print('Sampling model {}'.format(modelnum))
    data = torch.load(run_folder / f'model-{modelnum}.pt', map_location=device, weights_only=True)

    adapted_dict = {k[6:]: v for k, v in data['model'].items() if k.startswith('model.')}

    model.load_state_dict(adapted_dict)
    model.eval()

    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 1000,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        device=device,
        mode=mode
    )

    samples = diffusion.sample(batch_size=num_samples).cpu().numpy()

    if save:
        fig, ax = plotgrid(samples, mode=mode)
        fig.savefig(str(sampled_images_folder / f'model_{modelnum:02d}.png'))
        plt.close(fig)
    else:
        for i, sample in enumerate(samples):
            Source(param3d=sample, pix=True).plot(f'Model {modelnum} - Sample {i+1}')

# %% Diffusion progress visualization (single model, return_all_timesteps=True)

# data = torch.load(run_folder / 'model-10.pt', map_location=device, weights_only=True)
# adapted_dict = {k[6:]: v for k, v in data['model'].items() if k.startswith('model.')}
# model.load_state_dict(adapted_dict)
# model.eval()
# diffusion = GaussianDiffusion(model, image_size=64, timesteps=1000, sampling_timesteps=1000, device=device, mode=mode)
# samples_all_t = diffusion.sample(batch_size=num_samples, return_all_timesteps=True).cpu().numpy()

# def diffusion_progress_plotter(sample_idx=0, save=False):
#     fig, ax = plt.subplots(3, 8, figsize=(16,6))
#     for i,j in enumerate([0,500,750,875,937,968,984,1000]):
#         im = ax[0,i].imshow(samples_all_t[sample_idx,j,0], vmin=samples_all_t[sample_idx,-1,0].min(), vmax=samples_all_t[sample_idx,-1,0].max(), cmap='hot')
#         ax[0,i].axes.get_xaxis().set_visible(False)
#         ax[0,i].axes.get_yaxis().set_visible(False)
#         im = ax[1,i].imshow(samples_all_t[3,j,1], vmin=samples_all_t[3,-1,1].min(), vmax=samples_all_t[3,-1,1].max(), cmap='seismic')
#         ax[1,i].axes.get_xaxis().set_visible(False)
#         ax[1,i].axes.get_yaxis().set_visible(False)
#         im = ax[2,i].imshow(samples_all_t[3,j,2], vmin=samples_all_t[3,-1,2].min(), vmax=samples_all_t[3,-1,2].max(), cmap='plasma')
#         ax[2,i].axes.get_xaxis().set_visible(False)
#         ax[2,i].axes.get_yaxis().set_visible(False)
#     plt.subplots_adjust(wspace=0.02, hspace=0.02)
#     if save:
#         plt.savefig('/home/kamo/resources/slitless/figures/basp2025_plots/diff_sampling_amooo.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

# %% EIS Data Loading — distribution comparison
import glob

dataset_path = glob.glob('/home/kamo/resources/slitless/data/eis_data/datasets/dset_v2/')[0]
dbsnr = 100
noise_model = 'poisson'

ds = BasicDataset(data_dir=dataset_path, fold='train', dbsnr=dbsnr, noise_model=noise_model, numdetectors=3)
dl = DataLoader(ds, batch_size=len(ds), shuffle=True, num_workers=8)
y, x = next(iter(dl))
x = x.cpu().numpy()
means_eis = x.mean(axis=(0, 2, 3))
stds_eis = x.std(axis=(0, 2, 3))
maxs_eis = x.max(axis=(0, 2, 3))
mins_eis = x.min(axis=(0, 2, 3))

means_ddpm = samples.mean(axis=(0, 2, 3))
stds_ddpm = samples.std(axis=(0, 2, 3))
maxs_ddpm = samples.max(axis=(0, 2, 3))
mins_ddpm = samples.min(axis=(0, 2, 3))

print(f'Means EIS: {means_eis}')
print(f'Means DDPM: {means_ddpm}')
print(f'Stds EIS: {stds_eis}')
print(f'Stds DDPM: {stds_ddpm}')
print(f'Maxs EIS: {maxs_eis}')
print(f'Maxs DDPM: {maxs_ddpm}')
print(f'Mins EIS: {mins_eis}')
print(f'Mins DDPM: {mins_ddpm}')

# Save or display the EIS samples
for i in range(num_samples):
    Source(param3d=x[i], pix=True).plot(f'EIS {i+1}')

# # %% Plot Histograms
# plt.figure()
# plt.hist(x[:,0].flatten(), bins=100, color='orange', label='EIS')
# plt.title('EIS Intensity')
# xlims = plt.xlim()
# plt.show()

# plt.figure()
# plt.hist(samples[:,0].flatten(), bins=100, color='blue', label='DDPM')
# plt.title('DDPM Intensity')
# plt.xlim(xlims)
# plt.show()

# plt.figure()
# plt.hist(x[:,1].flatten(), bins=100, color='orange', range=(-0.5, 0.5))
# plt.title('EIS Velocity')
# plt.show()

# plt.figure()
# plt.hist(samples[:,1].flatten(), bins=100, color='blue', range=(-0.5, 0.5))
# plt.title('DDPM Velocity')
# plt.show()

# plt.figure()
# plt.hist(x[:,2].flatten(), bins=100, color='orange')
# plt.title('EIS Linewidth')
# xlims = plt.xlim()
# plt.show()

# plt.figure()
# plt.hist(samples[:,2].flatten(), bins=100)
# plt.title('DDPM Linewidth')
# plt.xlim(xlims)
# plt.show()
