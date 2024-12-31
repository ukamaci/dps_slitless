import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import matplotlib.pyplot as plt
from slitless.data_loader import BasicDataset
from torch.utils.data import DataLoader
from slitless.forward import Source, Imager, forward_op_torch
import numpy as np

path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
data = 'eis_train_5_64x64.npy' # 5 of 64x64 EIS dataset train images
rec_mode = 'all'

param4dar = np.load(path_data+data)

M = param4dar.shape[-1]
numdetectors = 3
dbsnr = 15
noise_model='poisson'
# noise_model='gaussian'

Sr = Source(
    param3d=torch.tensor(param4dar),
    # param3d=torch.tensor(param4dar[[0]]),
    pix=True
)
# mask= np.ones_like(Sr.param3d)

# imager = Imager(pixelated=True, mask=mask, dbsnr=dbsnr, max_count=dbsnr**2/0.9,
imager = Imager(pixelated=True, dbsnr=dbsnr, max_count=dbsnr**2/0.9,
noise_model=noise_model, spectral_orders=[0,-1,1,-2,2][:numdetectors])

imager.srpix = Sr
meas = imager.get_measurements()

channels = 3
# true = torch.tensor(param4dar[[0]])
true = torch.tensor(param4dar)


# %% DDPM Sampling
# Load the pretrained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def forward_op(x, device=None):
    if rec_mode == 'int':
        return torch.nn.Identity()(x)
    else:
        return forward_op_torch(true_intensity=x[:,0], true_doppler=x[:,1], true_linewidth=x[:,2], device=device)

model = Unet(
    channels=channels,
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
).to(device)

modellist = [1,5,10,20,30,40,50,60,70]
modellist = np.concatenate((np.array([1]), np.linspace(5,70,14).astype(int)))
rmses_list = []
for modelnum in modellist:
    data = torch.load('/home/kamo/resources/denoising-diffusion-pytorch/results/model-{}.pt'.format(modelnum), map_location=device, weights_only=True)

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
        recon = True,
        measurement = torch.tensor(meas).to(device),
        true=true.to(device),
        beta_schedule='cosine',
        # grad_scale = torch.tensor([1]).to(device),
        grad_scale = torch.tensor([1,1,1]).to(device),
        forward_op=forward_op,
        device=device,
        mode = rec_mode
    )

    # Generate new samples
    num_samples = 5
    samples, norms, grad_norms, rmses = diffusion.sample(batch_size=num_samples)
    samples = samples.detach().cpu().numpy()
    rmses = np.array(rmses).squeeze()
    if len(rmses.shape) == 3:
        rmses = rmses.mean(axis=1)

    truth_phy = imager.frompix(true.detach().cpu().numpy(), width_unit='km/s', array=True)
    recon_phy = imager.frompix(samples, width_unit='km/s', array=True)  
    rmse = np.sqrt(np.mean((truth_phy - recon_phy)**2, axis=(-1,-2)))
    rmses_list.append(rmse.mean(axis=0))

rmses_list = np.array(rmses_list)
plt.figure()
plt.plot(modellist, rmses_list[:,2], '-o')
plt.grid(which='both', axis='both')
plt.title('RMSE Width vs results/modelnum')
plt.show()
plt.savefig('rmses_width_vs_results_modelnum_db15.png')

plt.figure()
plt.plot(modellist, rmses_list[:,1], '-o')
plt.grid(which='both', axis='both')
plt.title('RMSE Vel vs results/modelnum')
plt.show()
plt.savefig('rmses_vel_vs_results_modelnum_db15.png')

plt.figure()
plt.plot(modellist, rmses_list[:,0], '-o')
plt.grid(which='both', axis='both')
plt.title('RMSE Int vs results/modelnum')
plt.show()
plt.savefig('rmses_int_vs_results_modelnum_db15.png')