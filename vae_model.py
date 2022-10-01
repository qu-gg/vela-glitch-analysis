"""
@file vae_modedl.py
@author Ryan Missel

Handles the model implementation code for the Variational AutoEncoder and the Denoising AutoEncoder
that was tested as an ablation in the original work
"""
import torch
import torch.nn as nn


class Flatten(nn.Module):
    """ Flattening module to use in Sequential blocks from arbitrary size to [BatchSize, -1] """
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """ Unflattening module to use in Sequential blocks from arbitrary size to [BatchSize, size, -1] """
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1)


class VAE(nn.Module):
    def __init__(self, device, image_channels=1, x_dim=200, h_dim=1280, z_dim=5):
        """
        Variational Auto-encoder network, used for reconstructing and denoising the pulsar signals effecitive
        :arg device: which device (GPU or CPU) to run on
        :arg image_channels: how many channels of the signal are present (1 in this work)
        :arg x_dim: length of the input signal
        :arg h_dim: bit adhoc, but dimension of the encoder output and decoder input
        :arg z_dim: size of the latent vector
        """
        super(VAE, self).__init__()
        self.device = device

        # Encoder block of the VAE, stacked 1D CNN
        self.encoder = nn.Sequential(
            nn.Conv1d(image_channels, 16, kernel_size=4, stride=2),
            nn.LeakyReLU(.1),
            nn.Conv1d(16, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(.1),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(.1),
            nn.Conv1d(64, 128, kernel_size=4, stride=2),
            nn.LeakyReLU(.1),
            Flatten()
        )

        # Linear layers to get distributional parameters of VAE embedding
        self.mu_layer = nn.Linear(h_dim, z_dim)
        self.logvar_layer = nn.Linear(h_dim, z_dim)

        # Decoding network, starting with a linear layer and then transposed 1D CNN layers to output space
        # Ends with an adaptive average pooling layer to go from CNN output to 200 timesteps
        self.latent_decode_layer = nn.Linear(z_dim, h_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose1d(32, 16, kernel_size=6, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose1d(16, image_channels, kernel_size=6, stride=2),
            nn.AdaptiveAvgPool1d(x_dim)
        )

    def reparameterize(self, mu, logvar):
        """ Re-parameterization trick for VAE sampling, allowing for backpropagation """
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        """ VAE information bottleneck and sampling """
        mu, logvar = self.mu_layer(h), self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        """ Encoding block with bottleneck and sampling """
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, h, mu, logvar

    def decode(self, z):
        """ Decoding block from latent to signal """
        z = self.latent_decode_layer(z)
        z = z.unsqueeze(1)
        z = self.decoder(z)
        return z.squeeze()

    def forward(self, x):
        """ Forward function, encodes and decodes """
        x = x.unsqueeze(1)
        z, h, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, h, mu, logvar

    def recon_from_latent(self, h):
        """
        Given a latent vector, handles the reparameterization and decoding to get the reconstructed sample
        :param h: latent vector from enc
        :return: reconstructed x
        """
        mu, logvar = self.mu_layer(h), self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z


class DAE(nn.Module):
    def __init__(self, device, denoise=False, image_channels=1, x_dim=200, h_dim=1280, z_dim=5):
        super(DAE, self).__init__()

        self.denoise = denoise
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv1d(image_channels, 16, kernel_size=4, stride=2),
            nn.LeakyReLU(.1),
            nn.Conv1d(16, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(.1),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(.1),
            nn.Conv1d(64, 128, kernel_size=4, stride=2),
            nn.LeakyReLU(.1),
            Flatten()
        )

        self.latent_in = nn.Linear(h_dim, z_dim)
        self.latent_out = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose1d(32, 16, kernel_size=6, stride=2),
            nn.LeakyReLU(.1),
            nn.ConvTranspose1d(16, image_channels, kernel_size=6, stride=2),
            nn.AdaptiveAvgPool1d(x_dim)
        )

    def encode(self, x):
        if self.denoise:
            x = x + torch.randn_like(x) * 0.3
        h = self.encoder(x)
        z = self.latent_in(h)
        return z, h

    def decode(self, z):
        z = self.latent_out(z)
        z = z.unsqueeze(1)
        z = self.decoder(z)
        return z.squeeze()

    def forward(self, x):
        x = x.unsqueeze(1)
        z, h = self.encode(x)
        z = self.decode(z)
        return z, None, None, None

    def recon_from_latent(self, h):
        """
        Given a latent vector, handles the reparameterization and decoding to get the reconstructed sample
        :param h: latent vector from enc
        :return: reconstructed x
        """
        mu, logvar = self.mu_layer(h), self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z