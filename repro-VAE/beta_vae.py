import torch
from base import BaseVAE
from torch import nn
from torch.nn import functional as F

class BetaVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self, in_channels: int,
                       latent_dim: int,
                       hidden_dims: list=None,
                       beta: float = 4,
                       gamma: float = 1000,
                       max_capacity: int = 25,
                       Capacity_max_iter: int = 1e5,
                       loss_type: str = 'B',
                       tau: float = 50.0,
                       **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        print('beta:', self.beta)
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.tau = tau

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder

        # Build Encoder with mean pooling
        for h_dim in hidden_dims:
            print(in_channels, h_dim)
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                            kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(h_dim),
                    nn.AvgPool2d(kernel_size=2),
                    nn.LeakyReLU())
            )
            in_channels = h_dim


        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 16, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 16, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            print(hidden_dims[i], hidden_dims[i+1])
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                    # nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU())
            )
          
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   hidden_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                # nn.BatchNorm2d(hidden_dims[-1]),
                nn.ReLU(),
                nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
                # nn.Tanh()
                )

    def encode(self, input):
        """
        Encodes the input by passing throught encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components --> in the lower bond, there is one part:
        # -Dkl(q(z|x) || p(z)). We assume p(z)~N(0,1) and use the encoder to predict the real
        # gaussian distribution of z.
        mu = self.fc_mu(result)   # mean
        log_var = self.fc_var(result)  # log of variance (make value can be positive and negative)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 256, 4, 4)  # ??? how to get 2, 2, is it the down-sampled image size
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough to compute the expectation
        for the loss?
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Log of Square of standard deviation of the latent Gaussian
        :return:
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu
    '''
    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        return [self.decode(z), input, mu, log_var]
    '''
    
    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z_decode = self.decode(z)
        mu_new, log_var_new = self.encode(z_decode)
        
        return [z_decode, input, mu, log_var, mu_new, log_var_new]

    def loss_function(self, *args, **kwargs):
        self.num_iter += 1
        recons = args[0][0]
        input_ = args[0][1]
        mu = args[0][2]
        log_var = args[0][3]
        mu_new = args[0][4]
        log_var_new = args[0][5]
        
        kld_weight = kwargs['M_N']  # Account for the minibatch from the data

        recons_loss = F.mse_loss(recons, input_, reduction='sum').div(kld_weight)   # reconstruction error, straightforward

        # KL Divergence: Lb = Eq(z|x)[logP(x|z)] - Dkl(q(z|x) || p(z))
        # Hope to increase the lower bond. So maximize the first part and minimize the second part
        # Dkl(q(z|x) || p(z)) leads to minimize sum(mu**2 + exp(log_var) - (1 + log_var))
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        sd1 = log_var.exp()
        sd2 = log_var_new.exp()
        p = torch.distributions.Normal(mu, sd1)
        q = torch.distributions.Normal(mu_new, sd2)
        kld_loss_pq = torch.distributions.kl_divergence(p, q).mean()

        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_loss + self.tau * kld_loss_pq
        elif self.loss_type == 'B':   # set beta large first (if beta is large, the recons_loss will be largetoo.
                                      # and then slowly reduce it. Introduce value C to control the beta and slowly
                                      # increase C.
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type')
        return {'loss': loss, "Reconstruction_loss": recons_loss, "KLD": kld_loss, "KLD_revise": kld_loss_pq}

    def sample(self, num_samples, current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding image space map
        Which can be used to reconstruct an image from the latent space (unnecessary for scene detection)
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate_image(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return:
        """
        return self.forward(x)[0]

    def generate_latent(self, x):
        """
        Given an input x, returns the corresponding latent space
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x latent_dim]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z