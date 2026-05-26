import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import matplotlib.pyplot as plt
from slitless.data_loader import BasicDataset
from torch.utils.data import DataLoader
from slitless.forward import Source, Imager, forward_op_torch
import numpy as np
# torch.manual_seed(0)

path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
data = 'eis_train_5_64x64.npy' # 5 of 64x64 EIS dataset train images
# data = 'eis_5_64x64.npy' # 5 of 64x64 EIS dataset train images
rec_mode = 'all'
suffix = '' if rec_mode=='all' else '_{}'.format(rec_mode)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

param4dar = np.load(path_data+data)

M = param4dar.shape[-1]
numdetectors = 3
dbsnr = 50
# noise_model='poisson'
noise_model='gaussian'
num_samples = 10

Sr = Source(
    # param3d=torch.tensor(param4dar),
    param3d=torch.tensor(param4dar[[1]]).to(device),
    pix=True
)
# mask= np.ones_like(Sr.param3d)

# imager = Imager(pixelated=True, mask=mask, dbsnr=dbsnr, max_count=dbsnr**2/0.9,
imager = Imager(pixelated=True, dbsnr=dbsnr, max_count=dbsnr**2/0.9,
noise_model=noise_model, spectral_orders=[0,-1,1,-2,2][:numdetectors])

imager.srpix = Sr
meas = imager.get_measurements()

# %% DDPM Sampling
# torch.manual_seed(0)
if rec_mode == 'int':
    meas = meas[:,[0]]
    channels = 1
    true = Sr.param3d[:,[0]]
elif rec_mode == 'vel':
    channels = 1
    true = Sr.param3d[:,[1]]
else:
    channels = 3
    # true = torch.tensor(param4dar[[0]])
    true = Sr.param3d


# Load the pretrained model

def forward_op(x, device=None):
    if rec_mode == 'int':
        return torch.nn.Identity()(x)
    elif rec_mode == 'vel':
        return forward_op_torch(
            true_intensity=Sr.param3d[:,0].repeat(x.shape[0],1,1), 
            true_doppler=x[:,0], 
            true_linewidth=Sr.param3d[:,2].repeat(x.shape[0],1,1), 
            device=device)
    else:
        return forward_op_torch(true_intensity=x[:,0], true_doppler=x[:,1], true_linewidth=x[:,2], device=device)

model = Unet(
    channels=channels,
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
).to(device)

data = torch.load('/home/kamo/resources/denoising-diffusion-pytorch/results{}/model-10.pt'.format(suffix), map_location=device, weights_only=True)

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
    measurement=torch.tensor(meas).to(device),
    true=true.to(device),
    beta_schedule='cosine',
    grad_scale = torch.tensor([1]).to(device),
    # grad_scale=1*torch.tensor([1,3.5,0.1]).to(device),
    # grad_scale=1.25*torch.tensor([1,1,1]).to(device),
    forward_op=forward_op,
    device=device,
    mode=rec_mode
)

# Generate new samples
samples, norms, grad_norms, rmses = diffusion.sample(batch_size=num_samples)
samples = samples.detach().cpu().numpy()
true = true.detach().cpu().numpy()
rmses = np.array(rmses).squeeze()
if len(rmses.shape) == 3:
    rmses = rmses.mean(axis=1)

plt.figure()
# plt.plot(torch.tensor(norms).detach().cpu().numpy())
plt.plot(norms)
plt.title('Norms')
plt.grid(which='both', axis='both')
plt.show()

plt.figure()
# plt.plot(torch.tensor(grad_norms).detach().cpu().numpy())
plt.plot(grad_norms)
plt.title('Grad Norms')
plt.grid(which='both', axis='both')
plt.show()

plt.figure()
plt.semilogy(rmses)
plt.legend(['int','vel','width'])
plt.title('RMSEs')
plt.grid(which='both', axis='both')
plt.show()

if rec_mode == 'all':
    truth_phy = imager.frompix(true, width_unit='km/s', array=True)
    recon_phy = imager.frompix(samples.mean(axis=0), width_unit='km/s', array=True)  
    recon_phy2 = imager.frompix(samples, width_unit='km/s', array=True)  
    rmse = np.sqrt(np.mean((truth_phy - recon_phy)**2, axis=(-1,-2)))
    rmse2 = np.sqrt(np.mean((truth_phy - recon_phy2)**2, axis=(-1,-2)))
    if true.shape[0]==1:
        print('rmse: {}'.format(rmse))
    print('rmse_all: {}'.format(rmse2))
    print('rmse_all_m: {}'.format(rmse2.mean(axis=0)))

    if true.shape[0]>1:
        for i in range(true.shape[0]):
            Sr.plot(idx=i, title='True {}'.format(i+1))
            Source(param3d=samples[i], pix=True).plot('Recon {}'.format(i+1))
    else:
        Sr.plot('True')
        for i in range(num_samples):
            Source(param3d=samples[i], pix=True).plot('Recon {}'.format(i+1))
        Source(param3d=samples.mean(axis=0), pix=True).plot('Recon Mean')
        Source(param3d=samples.std(axis=0), pix=True).plot('Recon Mean')


    recs = [meas[0], true[0], samples[0], samples[1], samples[2], samples.mean(axis=0)] 
    titles = ['Meas', 'True', 'Sample 1', 'Sample 2', 'Sample 3', 'Posterior Mean']

    vmin_row0 = recs[1][0].min()
    vmax_row0 = recs[1][0].max()
    vmin_row1 = recs[1][1].min()
    vmax_row1 = recs[1][1].max()
    vmin_row2 = recs[1][2].min()
    vmax_row2 = recs[1][2].max()

    fig, ax = plt.subplots(3, 6, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1]})
    # First row
    im = ax[0, 0].imshow(recs[0][0], cmap='hot', vmin=vmin_row0, vmax=vmax_row0)
    ax[0, 0].axes.get_xaxis().set_visible(False)
    ax[0, 0].axes.get_yaxis().set_visible(False)
    # ax[0, i].set_title(titles[i])

    # Second row
    im = ax[1, 0].imshow(recs[0][1], cmap='hot', vmin=vmin_row0, vmax=vmax_row0)
    ax[1, 0].axes.get_xaxis().set_visible(False)
    ax[1, 0].axes.get_yaxis().set_visible(False)

    # Third row
    im = ax[2, 0].imshow(recs[0][2], cmap='hot', vmin=vmin_row0, vmax=vmax_row0)
    ax[2, 0].axes.get_xaxis().set_visible(False)
    ax[2, 0].axes.get_yaxis().set_visible(False)

    for i in np.arange(1,6):
        # First row
        im = ax[0, i].imshow(recs[i][0], cmap='hot', vmin=vmin_row0, vmax=vmax_row0)
        ax[0, i].axes.get_xaxis().set_visible(False)
        ax[0, i].axes.get_yaxis().set_visible(False)
        # ax[0, i].set_title(titles[i])

        # Second row
        im = ax[1, i].imshow(recs[i][1], cmap='seismic', vmin=vmin_row1, vmax=vmax_row1)
        ax[1, i].axes.get_xaxis().set_visible(False)
        ax[1, i].axes.get_yaxis().set_visible(False)

        # Third row
        im = ax[2, i].imshow(recs[i][2], cmap='plasma', vmin=vmin_row2, vmax=vmax_row2)
        ax[2, i].axes.get_xaxis().set_visible(False)
        ax[2, i].axes.get_yaxis().set_visible(False)

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.895, 0.713, 0.01, 0.267])  # Adjust the position and size
    cbar_ax2 = fig.add_axes([0.895, 0.368, 0.01, 0.267])
    cbar_ax3 = fig.add_axes([0.895, 0.020, 0.01, 0.267])

    fig.colorbar(ax[0, 1].images[0], cax=cbar_ax1, orientation='vertical')
    fig.colorbar(ax[1, 1].images[0], cax=cbar_ax2, orientation='vertical')
    fig.colorbar(ax[2, 1].images[0], cax=cbar_ax3, orientation='vertical')

    plt.tight_layout(h_pad=4, rect=[0, 0, 0.9, 1])  # Leave space for colorbars
    plt.show()

    # recs = [true[0], samples[0], samples.mean(axis=0), samples.std(axis=0), abs(samples.mean(axis=0)-true[0])] 
    # titles = ['True', 'Point Est', 'MMSE', 'STD', 'Error']

    # vmin_row0 = recs[0][0].min()
    # vmax_row0 = recs[0][0].max()
    # vmin_row1 = recs[0][1].min()
    # vmax_row1 = recs[0][1].max()
    # vmin_row2 = recs[0][2].min()
    # vmax_row2 = recs[0][2].max()

    # fig, ax = plt.subplots(3, 5, figsize=(11, 7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1]})

    # for i in range(5):
    #     # First row
    #     im = ax[0, i].imshow(
    #         recs[i][0], cmap='hot', vmin=vmin_row0 if i < 3 else None, vmax=vmax_row0 if i < 3 else None
    #     )
    #     ax[0, i].axes.get_xaxis().set_visible(False)
    #     ax[0, i].axes.get_yaxis().set_visible(False)
    #     ax[0, i].set_title(titles[i])

    #     # Second row
    #     im = ax[1, i].imshow(
    #         recs[i][1], cmap='seismic', vmin=vmin_row1 if i < 3 else None, vmax=vmax_row1 if i < 3 else None
    #     )
    #     ax[1, i].axes.get_xaxis().set_visible(False)
    #     ax[1, i].axes.get_yaxis().set_visible(False)

    #     # Third row
    #     im = ax[2, i].imshow(
    #         recs[i][2], cmap='plasma', vmin=vmin_row2 if i < 3 else None, vmax=vmax_row2 if i < 3 else None
    #     )
    #     ax[2, i].axes.get_xaxis().set_visible(False)
    #     ax[2, i].axes.get_yaxis().set_visible(False)

    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout for colorbars
    # plt.show()

else:
    if rec_mode == 'int':
        cmap='hot'
        factor = 1
    elif rec_mode == 'vel':
        cmap='seismic'
        factor = 34.2483
    elif rec_mode == 'width':
        cmap='plasma'
    rmse = np.sqrt(np.mean((true - samples)**2, axis=(-1,-2)))*factor
    rmse_meas = np.sqrt(np.mean((true - meas.cpu().numpy())**2, axis=(-1,-2)))*factor
    rmse2 = np.sqrt(np.mean((true - samples.mean(axis=0))**2, axis=(-1,-2)))*factor


    if true.shape[0]>1:
        for i in range(true.shape[0]):
            plt.figure()
            plt.imshow(true[i].squeeze(), cmap=cmap)
            plt.title('True {}'.format(i+1))
            plt.colorbar()

            plt.figure()
            plt.imshow(samples[i].squeeze(), cmap=cmap)
            plt.title('Recon {}'.format(i+1))
            plt.colorbar()

        plt.figure()
        plt.imshow(samples.mean(axis=0).squeeze(), cmap=cmap)
        plt.title('Recon MMSE')
        plt.colorbar()

    else:
        plt.figure()
        plt.imshow(true.squeeze(), cmap=cmap)
        plt.title('True')
        plt.colorbar()

        plt.figure()
        plt.imshow(samples.mean(axis=0).squeeze(), cmap=cmap)
        plt.title('Recon MMSE')
        plt.colorbar()

        for i in range(num_samples):
            plt.figure()
            plt.imshow(samples[i].squeeze(), cmap=cmap)
            plt.title('Recon {}'.format(i+1))
            plt.colorbar()

    print('rmse: {}'.format(rmse))
    if true.shape[0]>1:
        print('rmse_mean: {}'.format(rmse.mean(axis=0)))
    else:
        print('rmse_all: {}'.format(rmse2))
    print('rmse_meas: {}'.format(rmse_meas))

# %%
