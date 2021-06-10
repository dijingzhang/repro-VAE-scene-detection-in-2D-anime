import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='11785_Project')

    parser.add_argument('--max_iters', type=int, default=100, help='the number of epochs for training')
    parser.add_argument('--train_folder', type=str, default='/content/gdrive/MyDrive/11785-project/project-dataset/train_data/')
    parser.add_argument('--val_folder', type=str, default='/content/gdrive/MyDrive/11785-project/project-dataset/val_data/')

    # hyperparameters for optimizer
    parser.add_argument('--lr', type=float, default=0.1, help='the learning rate for training')
    parser.add_argument('--w_decay', type=float, default=5e-4, help='weight decay for optimizer')
    parser.add_argument('--m', type=float, default=0.5, help='momentum for optimizer')
    parser.add_argument('--bs', type=int, default=256, help='batch size')

    # hyperparameters for beta-vae
    parser.add_argument('--latent_dim', type=int, default=1024, help='the dimension of latent space')
    parser.add_argument('--beta', type=float, default=4, help='beta value for beta-vae')
    parser.add_argument('--gamma', type=float, default=1000.0, help='gamma value for modified beta-vae')
    parser.add_argument('--max_capacity', type=int, default=25)
    parser.add_argument('--Capacity_max_iter', type=int, default=1e5)
    parser.add_argument('--loss_type', type=str, default='H')
    parser.add_argument('--hidden_dims', type=list, default=[32, 64, 128, 256])
    parser.add_argument('--output_folder', type=str, default='/content/gdrive/MyDrive/11785-project/reprojection_100')
    parser.add_argument('--tau', type=float, default=50)
    opts = parser.parse_args()

    return opts