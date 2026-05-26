import matplotlib.pyplot as plt
import numpy as np
import torch

def plotgrid(arr, mode='all'):
    if type(arr) == torch.Tensor:
        arr = arr.detach().cpu().numpy()
    if mode == 'all':
        fig, ax = plt.subplots(3, 8, figsize=(20, 8))
        for i in range(min(8, arr.shape[0])):
            ax[0,i].imshow(arr[i,0], cmap='hot')
            ax[1,i].imshow(arr[i,1], cmap='seismic')
            ax[2,i].imshow(arr[i,2], cmap='plasma')
        plt.tight_layout()
    elif mode == 'int':
        fig, ax = plt.subplots(3, 5, figsize=(15, 10))
        for i in range(min(15, arr.shape[0])):
            ax[i//5,i%5].imshow(arr[i].squeeze(), cmap='hot')
        plt.tight_layout()
    elif mode == 'vel':
        fig, ax = plt.subplots(3, 5, figsize=(15, 10))
        for i in range(min(15, arr.shape[0])):
            ax[i//5,i%5].imshow(arr[i].squeeze(), cmap='seismic')
        plt.tight_layout()
    elif mode == 'width':
        fig, ax = plt.subplots(3, 5, figsize=(15, 10))
        for i in range(min(15, arr.shape[0])):
            ax[i//5,i%5].imshow(arr[i].squeeze(), cmap='plasma')
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

                
