import matplotlib.pyplot as plt
import numpy as np
import torch

SPEEDOFLIGHT = 299792.458   # km/s
WAVELENGTH   = 195.117937907451  # Å  (Fe XII)
WIDTH_KMS    = SPEEDOFLIGHT / WAVELENGTH  # Å → km/s conversion factor

def plotgrid(arr, mode='all'):
    if type(arr) == torch.Tensor:
        arr = arr.detach().cpu().numpy()
    if mode == 'all':
        fig, ax = plt.subplots(3, 8, figsize=(20, 8))
        for i in range(min(8, arr.shape[0])):
            im0 = ax[0,i].imshow(arr[i,0], cmap='hot')
            im1 = ax[1,i].imshow(arr[i,1], cmap='seismic')
            im2 = ax[2,i].imshow(arr[i,2] * WIDTH_KMS, cmap='plasma')
            for im, a in zip([im0, im1, im2], ax[:,i]):
                fig.colorbar(im, ax=a, orientation='horizontal', location='top', pad=0.02, fraction=0.046)
    elif mode == 'int':
        fig, ax = plt.subplots(3, 5, figsize=(15, 10))
        for i in range(min(15, arr.shape[0])):
            im = ax[i//5,i%5].imshow(arr[i].squeeze(), cmap='hot')
            fig.colorbar(im, ax=ax[i//5,i%5], orientation='horizontal', location='top', pad=0.02, fraction=0.046)
    elif mode == 'vel':
        fig, ax = plt.subplots(3, 5, figsize=(15, 10))
        for i in range(min(15, arr.shape[0])):
            im = ax[i//5,i%5].imshow(arr[i].squeeze(), cmap='seismic')
            fig.colorbar(im, ax=ax[i//5,i%5], orientation='horizontal', location='top', pad=0.02, fraction=0.046)
    elif mode == 'width':
        fig, ax = plt.subplots(3, 5, figsize=(15, 10))
        for i in range(min(15, arr.shape[0])):
            im = ax[i//5,i%5].imshow(arr[i].squeeze() * WIDTH_KMS, cmap='plasma')
            fig.colorbar(im, ax=ax[i//5,i%5], orientation='horizontal', location='top', pad=0.02, fraction=0.046)

    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()

    return fig, ax

if __name__ == '__main__':
    path_data = '/home/kamo/resources/slitless/data/datasets/baseline/'
    data = 'eis_5_64x64.npy' # 5 of 64x64 EIS dataset train images
    # data = 'eis_train_5_64x64.npy' # 5 of 64x64 EIS dataset train images

    param4dar = np.load(path_data+data)
    param4dar2 = np.zeros((8,3,64,64))
    param4dar2[:5] = param4dar[:]
    param4dar2[5:] = param4dar2[:3]

    mode = 'all'
    fig, ax = plotgrid(param4dar2, mode=mode)
    # fig, ax = plotgrid(param4dar[:,[2]], mode=mode)
    # fig.savefig('/home/kamo/resources/denoising-diffusion-pytorch/training_results/<run>/sampled_images/eis_train.png')
    # fig.savefig('/home/kamo/resources/denoising-diffusion-pytorch/training_results/<run>/sampled_images/eis_test.png')

                
