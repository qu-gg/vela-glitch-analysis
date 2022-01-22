import torch
import torch.nn as nn


def init_weights(weights):
    for layer in weights:
        nn.init.normal_(layer.weight.data, 0, 0.001)
        if layer.bias is not None:
            layer.bias.data.zero_()


def init_weights_(layer):
    nn.init.normal_(layer.weight.data, 0, 0.001)
    if layer.bias is not None:
        layer.bias.data.zero_()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1)


class VAE(nn.Module):
    def __init__(self, device, image_channels=1, x_dim=200, h_dim=1280, z_dim=5):
        super(VAE, self).__init__()

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

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

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

        # self.encoder.apply(init_weights_)
        # init_weights([self.fc1, self.fc2, self.fc3])
        # self.decoder.apply(init_weights_)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        #         print(h.shape)
        z, mu, logvar = self.bottleneck(h)
        return z, h, mu, logvar

    def decode(self, z):
        #         print(z.shape)
        z = self.fc3(z)
        z = z.unsqueeze(1)
        #         print(z.shape)
        z = self.decoder(z)
        return z.squeeze()

    def forward(self, x):
        #         print(x.shape)
        x = x.unsqueeze(1)
        #         print(x.shape)
        z, h, mu, logvar = self.encode(x)
        #         print(z.shape)
        z = self.decode(z)
        return z, h, mu, logvar

    def recon_from_latent(self, h):
        """
        Given a latent vector, handles the reparameterization and decoding to get the reconstructed sample
        :param h: latent vector from enc
        :return: reconstructed x
        """
        mu, logvar = self.fc1(h), self.fc2(h)
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

        # self.encoder.apply(init_weights_)
        # init_weights([self.fc1, self.fc2, self.fc3])
        # self.decoder.apply(init_weights_)

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
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z