import torch
import torch.utils.data as data
from beta_vae import BetaVAE
import torch.optim as optim
import os
import numpy as np

def Train(model, train_dataset, val_dataset, batch_size, max_iters, lr, w_decay, m, output_folder):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nw = 2 if device != 'cpu' else 0
    print("Running Device is", device, " with num of workers ", nw)

    train_data_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=nw)
    val_data_loader = data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=nw)

    # initialize model
    Model = model
    Model.to(device)

    # Optimzier
    # optimizer = optim.SGD(Model.parameters(), lr=lr, weight_decay=w_decay, momentum=m)
    optimizer = optim.Adam(Model.parameters(), lr=lr, weight_decay=w_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)
    # save training loss and validation loss for KFold
    train_loss = []
    val_loss = []
    train_recons_loss = []
    train_kld_loss = []
    train_kld_revise_loss = []

    # Training
    for iter in range(max_iters):
        Model.train()
        loss = 0
        recons_loss = 0
        kld_loss = 0
        batch_num = 0
        kld_revise_loss = 0
        for images in train_data_loader:
            imgs = images.to(device)

            optimizer.zero_grad()
            output = Model.forward(imgs)

            loss_tmp = Model.loss_function(output, M_N=batch_size)
            
            # keep records of recons loss and kld loss as well
            loss += loss_tmp['loss'].item()
            recons_loss += loss_tmp['Reconstruction_loss'].item()
            kld_loss += loss_tmp['KLD'].item()
            kld_revise_loss += loss_tmp['KLD_revise'].item()
            # print('total:', loss_tmp['loss'].item(), 'kld loss:', loss_tmp['KLD'].item(), 'recons loss:', loss_tmp['Reconstruction_loss'].item())
            loss_tmp['loss'].backward()
            optimizer.step()
            # compute the number of batch
            batch_num += 1
            torch.cuda.empty_cache()
            del imgs

        print('Train: #{} epoch, the loss is {}, recons loss is {}, kld loss is {}, kld revise loss is {}'.format(iter, 
                                                        loss/batch_num, recons_loss/batch_num, kld_loss/batch_num, kld_revise_loss/batch_num))
        train_loss.append(loss / batch_num)
        train_recons_loss.append(recons_loss / batch_num)
        train_kld_loss.append(kld_loss / batch_num)
        train_kld_revise_loss.append(kld_revise_loss / batch_num)
        scheduler.step(loss/batch_num)

        # Validation
        # batch_num = 0
        # loss_val = 0
        # Model.eval()
        # with torch.no_grad():
        #     for images in val_data_loader:
        #         imgs = images.to(device)
        #         output = Model.forward(imgs)
        #         loss_val += Model.loss_function(output, M_N=batch_size)['loss'].item()
        #         batch_num += 1
        #         torch.cuda.empty_cache()
        #         del imgs

        # print('Val: #{} epoch, the loss is'.format(iter), loss_val / batch_num)
        # val_loss.append(loss_val / batch_num)
        
        output_model_path = os.path.join(output_folder, 'model_state_' + str(iter) + '_val_loss_' + str(loss/batch_num) + '.pkl')
        torch.save(Model.state_dict(), output_model_path)
        if iter % 10 == 9:
            loss_file = os.path.join(output_folder,'loss_iter_' + str(iter) + '.npy')
            loss_record = {'total_loss':train_loss, 'recons_loss':train_recons_loss, 'kld_loss':train_kld_loss}
            np.save(loss_file, loss_record)
    return Model.state_dict(), train_loss, val_loss

def Infer(model, val_dataset, batch_size, latent_dim, output_folder):
    
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
    # calculate KLD for each consecutive image pairs
    print(latent_dis_list[0])
    
    for i in range(len(latent_dis_list) - 1):
        mu_1, sd_1 = latent_dis_list[i]
        mu_2, sd_2 = latent_dis_list[i+1]

        p = torch.distributions.Normal(mu_1,sd_1)
        q = torch.distributions.Normal(mu_2,sd_2)

        KLD = torch.distributions.kl_divergence(p, q).mean()
        kld_list.append(KLD)


    # save kld list as npy file
    kld_file = os.path.join(output_folder,'kld_list.npy')
    np.save(kld_file, kld_list)
    return kld_list





