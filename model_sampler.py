import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.plotting import plotgrid
import matplotlib.pyplot as plt
from slitless.data_loader import BasicDataset
from torch.utils.data import DataLoader
from slitless.forward import Source
import numpy as np

# %% DDPM Sampling
mode = 'width'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels = 3 if mode == 'all' else 1
suffix = '' if mode=='all' else '_{}'.format(mode)

# Load the pretrained model
model = Unet(
    dim = 64,
    channels=channels,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
).to(device)

modellist = [1,2,3,4,5,6]
# modellist = [10,20,30,40,50]
# modellist = np.concatenate((np.array([1]), np.linspace(5,70,14).astype(int)))
rmses_list = []
for modelnum in modellist:
    print('Sampling model {}'.format(modelnum))
    data = torch.load('/home/kamo/resources/denoising-diffusion-pytorch/results{}/model-{}.pt'.format(suffix,modelnum), map_location=device, weights_only=True)

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
        device=device,
        mode=mode
    )

    # Generate new samples
    num_samples = 15
    samples = diffusion.sample(batch_size=num_samples).cpu().numpy()

    fig, ax = plotgrid(samples, mode=mode)
    fig.savefig('/home/kamo/resources/denoising-diffusion-pytorch/model_samples/{}/model_{:02d}.png'.format(mode,modelnum))
    