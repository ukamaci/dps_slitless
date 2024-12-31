import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import matplotlib.pyplot as plt
from slitless.data_loader import BasicDataset
from torch.utils.data import DataLoader
from slitless.forward import Source

# %% DDPM Sampling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pretrained model
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
).to(device)

data = torch.load('./results/model-40.pt', map_location=device, weights_only=True)

adapted_dict = {k[6:]: v for k, v in data['model'].items() if k.startswith('model.')}

model.load_state_dict(adapted_dict)

# model.load_state_dict(torch.load('./results/model-25.pt', weights_only=True))
model.eval()

# Initialize the diffusion process
diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 1000,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    device=device
)

# Generate new samples
num_samples = 1
samples = diffusion.sample(batch_size=num_samples).cpu().numpy()

# %% EIS Data Loading
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

# %% Plot samples

# Save or display the generated samples
for i, sample in enumerate(samples):
    Source(param3d=sample, pix=True).plot(f'DDPM {i+1}')

# Save or display the generated samples
for i in range(num_samples):
    Source(param3d=x[i], pix=True).plot(f'EIS {i+1}')

# %% Plot Histograms
plt.figure()
plt.hist(x[:,0].flatten(), bins=100, color='orange', label='EIS')
plt.title('EIS Intensity')
xlims = plt.xlim()
plt.show()

plt.figure()
plt.hist(samples[:,0].flatten(), bins=100, color='blue', label='DDPM')
plt.title('DDPM Intensity')
plt.xlim(xlims)
plt.show()

plt.figure()
plt.hist(x[:,1].flatten(), bins=100, color='orange', range=(-0.5, 0.5))
plt.title('EIS Velocity')
plt.show()

plt.figure()
plt.hist(samples[:,1].flatten(), bins=100, color='blue', range=(-0.5, 0.5))
plt.title('DDPM Velocity')
plt.show()

plt.figure()
plt.hist(x[:,2].flatten(), bins=100, color='orange')
plt.title('EIS Linewidth')
xlims = plt.xlim()
plt.show()

plt.figure()
plt.hist(samples[:,2].flatten(), bins=100)
plt.title('DDPM Linewidth')
plt.xlim(xlims)
plt.show()
