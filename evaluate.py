import torch
import glob
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import matplotlib.pyplot as plt
from slitless.data_loader import BasicDataset
from torch.utils.data import DataLoader
from einops import rearrange, repeat
import numpy as np
from scipy import linalg
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torch.nn.functional import adaptive_avg_pool2d
from statistics import NormalDist

def fid_score(real_samples, fake_samples, device="cuda"):
    """Calculates the FID score between real and generated samples."""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_v3 = InceptionV3([block_idx]).to(device)

    def get_features(samples):
        inception_v3.eval()
        features = inception_v3(samples)[0]
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features.cpu().numpy()

    # Get features
    real_features = get_features(real_samples.to(device).float())
    fake_features = get_features(fake_samples.to(device).float())

    # Calculate means and covariances
    m1, s1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    m2, s2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    # Calculate FID
    return calculate_frechet_distance(m1, s1, m2, s2)

def gaussmatch_score(means1, stds1, means2, stds2):
    scores = np.zeros(3)
    for i in range(3):
        scores[i] = NormalDist(mu=means1[i], sigma=stds1[i]).overlap(
            NormalDist(mu=means2[i], sigma=stds2[i]))
    return scores

# %% DDPM Sampling
# Load the pretrained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

samples = []
# modellist = [1,5,10,15,20,25]
modellist = [10,40,70]
num_samples = 100

# fid_scores = []
means_ddpm = []
stds_ddpm = []

dataset_path = glob.glob('/home/kamo/resources/slitless/data/eis_data/datasets/dset_v2/')[0]
dbsnr = 100
noise_model = 'poisson'

ds = BasicDataset(data_dir=dataset_path, fold='train', dbsnr=dbsnr, noise_model=noise_model, numdetectors=3)
dl = DataLoader(ds, batch_size=len(ds), shuffle=True, num_workers=8)
y, x = next(iter(dl))
x = x.cpu().numpy()

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
).to(device)

# Initialize the diffusion process
diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 1000,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    device=device
)

means_eis = x.mean(axis=(0, 2, 3))
stds_eis = x.std(axis=(0, 2, 3))
maxs_eis = x.max(axis=(0, 2, 3))
mins_eis = x.min(axis=(0, 2, 3))

gaussmatch_scores = []

for modelnum in modellist:

    data = torch.load('./results/model-{}.pt'.format(modelnum), map_location=device, weights_only=True)

    adapted_dict = {k[6:]: v for k, v in data['model'].items() if k.startswith('model.')}

    model.load_state_dict(adapted_dict)

    # model.load_state_dict(torch.load('./results/model-25.pt', weights_only=True))
    model.eval()

    # Generate new samples
    samples.append(diffusion.sample(batch_size=num_samples).cpu().numpy())
    # fid_scores.append(fid_score(torch.tensor(x[:10]), torch.tensor(samples[-1]), device=device))
    means_ddpm.append(samples[-1].mean(axis=(0, 2, 3)))
    stds_ddpm.append(samples[-1].std(axis=(0, 2, 3)))

    gaussmatch_scores.append(gaussmatch_score(means_eis, stds_eis, means_ddpm[-1], stds_ddpm[-1]))

means_ddpm = np.array(means_ddpm)
stds_ddpm = np.array(stds_ddpm)
gaussmatch_scores = np.array(gaussmatch_scores)

# %% Plot Histograms
# plt.figure()
# plt.plot(modellist, fid_scores, '-o')
# plt.title('FID Scores vs Model Number')
# plt.show()

# plt.figure()
# plt.plot(modellist, np.ones(len(modellist))*means_eis[0], '-o', label='EIS Int')
# plt.plot(modellist, means_ddpm[:,0], '-o', label='DDPM Int')
# plt.grid(which='both', axis='both')
# plt.title('Means Intensity')
# plt.show()

# plt.figure()
# plt.plot(modellist, np.ones(len(modellist))*means_eis[1], '-o', label='EIS Vel')
# plt.plot(modellist, means_ddpm[:,1], '-o', label='DDPM Vel')
# plt.grid(which='both', axis='both')
# plt.title('Means Velocity')
# plt.show()

# plt.figure()
# plt.plot(modellist, np.ones(len(modellist))*means_eis[2], '-o', label='EIS Width')
# plt.plot(modellist, means_ddpm[:,2], '-o', label='DDPM Width')
# plt.grid(which='both', axis='both')
# plt.title('Means Width')
# plt.show()

# plt.figure()
# plt.plot(modellist, np.ones(len(modellist))*stds_eis[0], '-o', label='EIS Int')
# plt.plot(modellist, stds_ddpm[:,0], '-o', label='DDPM Int')
# plt.grid(which='both', axis='both')
# plt.title('Stds Intensity')
# plt.show()

# plt.figure()
# plt.plot(modellist, np.ones(len(modellist))*stds_eis[1], '-o', label='EIS Vel')
# plt.plot(modellist, stds_ddpm[:,1], '-o', label='DDPM Vel')
# plt.grid(which='both', axis='both')
# plt.title('Stds Velocity')
# plt.show()

# plt.figure()
# plt.plot(modellist, np.ones(len(modellist))*stds_eis[2], '-o', label='EIS Width')
# plt.plot(modellist, stds_ddpm[:,2], '-o', label='DDPM Width')
# plt.grid(which='both', axis='both')
# plt.title('Stds Width')
# plt.show()

plt.figure()
plt.plot(modellist, gaussmatch_scores, '-o')
plt.legend(['int','vel','width'])
plt.grid(which='both', axis='both')
plt.title('Stds Width')
plt.show()