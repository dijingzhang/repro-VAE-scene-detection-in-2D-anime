import torch
from base import BaseVAE
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.utils.data as data
from beta_vae import BetaVAE
import torch.optim as optim
from sklearn.model_selection import KFold
from dataloader import *
from train import Train
from infer import Infer
from generate import *
from detect import Detect
from metric import *
import argparse
import os
from torchsummary import summary
def linearSearch(kld_list, top_K = 257):
    
    top_K = len(kld_list) if top_K > len(kld_list) else top_K
    kld_tuple_list = []
    for i in range(len(kld_list)):
        kld_tuple_list.append([kld_list[i], i])
    kld_tuple_list.sort(reverse=True)
    index_of_change = [temp[1] for temp in kld_tuple_list[:top_K]]
    return index_of_change

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='11785_Project_inference')
    parser.add_argument('--model', type=str, default='model.pth')
    parser.add_argument('--model_folder', type=str, default='/models/')
    parser.add_argument('--image_folder', type=str, default='/content/project/val_data/')
    parser.add_argument('--max_iters', type=int, default=100, help='the number of epochs for training')
    parser.add_argument('--span', type=int, default=1, help='span of model folder')
    # hyperparameters for optimizer
    parser.add_argument('--lr', type=float, default=0.1, help='the learning rate for training')
    parser.add_argument('--w_decay', type=float, default=5e-4, help='weight decay for optimizer')
    parser.add_argument('--m', type=float, default=0.5, help='momentum for optimizer')

    parser.add_argument('--bs', type=int, default=256, help='batch size')

    parser.add_argument('--threshold', type=float, default=0.5, help='the threshold for loss. Below it, training stops')

    # hyperparameters for beta-vae
    parser.add_argument('--latent_dim', type=int, default=1024, help='the dimension of latent space')
    parser.add_argument('--beta', type=int, default=4, help='beta value for beta-vae')
    parser.add_argument('--gamma', type=float, default=1000.0, help='gamma value for modified beta-vae')
    parser.add_argument('--max_capacity', type=int, default=25)
    parser.add_argument('--Capacity_max_iter', type=int, default=1e5)
    parser.add_argument('--loss_type', type=str, default='H')
    parser.add_argument('--hidden_dims', type=list, default=[32,64,128,256])
    parser.add_argument('--output_folder', type=str, default='/content/gdrive/MyDrive/infer_results/')

    # hyperparameters for inference
    parser.add_argument('--top_K', type=int, default=257, help='the number of scene change to detect')
    opts = parser.parse_args()

    csv_name = 'val_frame.csv'
    # save2csv(path=opts.image_folder, csvname=csv_name)
    Model = BetaVAE(in_channels=3, latent_dim=opts.latent_dim, hidden_dims=opts.hidden_dims, beta=opts.beta,
        gamma=opts.gamma, max_capacity=opts.max_capacity, Capacity_max_iter=opts.Capacity_max_iter, loss_type=opts.loss_type)


    
        
    # file_list = []
    print("changed")
    directory = opts.model_folder
    model_path_list = []
    for filename in os.listdir(directory):
        if filename.endswith('pkl'):
            # model_path_list.append(os.path.join(directory, filename))
            model_path_list.append(filename)
    print(model_path_list)
    model_path_list.sort(key=lambda x:int(x.split('_')[2]))
    
    # model_path_list.sort()
    for i in range(len(model_path_list)):
        if i < 29 or i > 49:
          continue
        if i % opts.span == opts.span-1:
        # if i > -1:

            print(model_path_list[i])

            model_state = torch.load(os.path.join(directory, model_path_list[i]))
            Model.load_state_dict(model_state)
            Model.to("cuda")
            summary(Model, (3, 64, 64))
            break
            dataset = Dataload(imgpath=opts.image_folder, csv_name=csv_name)
            with torch.no_grad():
                print("infer")
                kld_list = Infer(Model, dataset, batch_size=opts.bs, latent_dim=opts.latent_dim, output_folder=opts.model_folder)

            index_of_change = linearSearch(kld_list, opts.top_K)  

            np.save(os.path.join(opts.model_folder, 'scene_change_res' + str(i) + '.npy'), index_of_change)
       


