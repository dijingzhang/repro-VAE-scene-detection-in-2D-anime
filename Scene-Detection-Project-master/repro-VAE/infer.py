import torch
import torch.utils.data as data
from beta_vae import BetaVAE
import torch.optim as optim
import os
import numpy as np


def Infer(model, val_dataset, batch_size, latent_dim, output_folder):
    print("start to infer")
    # output_folder = '/content/gdrive/MyDrive/11785_project_latent_vectors_output/'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nw = 2 if device != 'cpu' else 0
    print("Running Device is", device, " with num of workers ", nw)

    val_data_loader = data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=nw)

    # initialize model
    Model = model
    Model.to(device)
    Model.eval()

    # save training loss and validation loss for KFold
    kld_list = []
    latent_dis_list = torch.zeros(2, len(val_dataset), latent_dim)
    print("shape of list:", latent_dis_list.shape)
    # Inference
    # get latent vectors for each image

    batch_ind = 0
    for images in val_data_loader:
        imgs = images.to(device)
        latent_distributions = Model.encode(imgs)
        latent_dis_list[0, batch_ind: batch_ind + latent_distributions[0].shape[0], :] = latent_distributions[0]
        latent_dis_list[1, batch_ind: batch_ind + latent_distributions[0].shape[0], :] = latent_distributions[1]
        batch_ind += latent_distributions[0].shape[0]
        print(batch_ind)
        del latent_distributions

    # calculate KLD for each consecutive image pairs
    print(latent_dis_list[0])
    
    for i in range(latent_dis_list.shape[1] - 1):
        mu_1, sd_1 = latent_dis_list[:,i,:]
        mu_2, sd_2 = latent_dis_list[:,i+1,:]

        p = torch.distributions.Normal(mu_1,torch.exp(0.5 * sd_1))
        q = torch.distributions.Normal(mu_2,torch.exp(0.5 * sd_2))

        KLD = torch.distributions.kl_divergence(p, q).mean()
        kld_list.append(KLD)


    # save kld list as npy file
    kld_file = os.path.join(output_folder,'kld_list.npy')
    np.save(kld_file, kld_list)
    return kld_list





