import math

import torch
from torch import nn
from torch.nn import functional as F


class Autoencoder(nn.Module):
    def __init__(self, sparse=False, latent_dim=64):
        super().__init__()
        self.sparse = sparse
        self.hidden = nn.Linear(28 * 28, 256)
        self.to_latent = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def encode_with_hidden(self, x):
        hidden = F.relu(self.hidden(x.flatten(1)))
        return hidden, F.relu(self.to_latent(hidden))

    def encode(self, x):
        return self.encode_with_hidden(x)[1]

    def decode(self, z):
        return self.decoder(z).reshape(-1, 1, 28, 28)

    def forward(self, x):
        return self.decode(self.encode(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.hidden = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.hidden(x)
        return self.mu(hidden), self.logvar(hidden)

    def decode(self, z):
        return self.decoder(z).reshape(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


class MLPGenerator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.network(z).reshape(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


class DCGenerator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.network(z.reshape(z.shape[0], z.shape[1], 1, 1))


class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 7, 1, 0),
        )

    def forward(self, x):
        return self.network(x).reshape(-1)


def timestep_embedding(timesteps, dim):
    half = dim // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time = nn.Linear(time_dim, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embedding):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(self.conv2(F.silu(h)))
        return F.silu(h + self.skip(x))


class SmallUNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.input = nn.Conv2d(1, 64, 3, padding=1)
        self.down1 = TimeBlock(64, 64, time_dim)
        self.down2 = TimeBlock(64, 128, time_dim)
        self.middle = TimeBlock(128, 128, time_dim)
        self.up1 = TimeBlock(256, 64, time_dim)
        self.up2 = TimeBlock(128, 64, time_dim)
        self.output = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, timesteps):
        time = self.time_mlp(timestep_embedding(timesteps, 128))
        first = self.down1(self.input(x), time)
        second = self.down2(F.avg_pool2d(first, 2), time)
        middle = self.middle(F.avg_pool2d(second, 2), time)
        up = F.interpolate(middle, size=second.shape[-2:], mode="nearest")
        up = self.up1(torch.cat((up, second), dim=1), time)
        up = F.interpolate(up, size=first.shape[-2:], mode="nearest")
        up = self.up2(torch.cat((up, first), dim=1), time)
        return self.output(up)


class MLPClassifier(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 10)

    def forward(self, x):
        return self.fc3(self.encode(x))

    def encode(self, x):
        x = x.flatten(1)
        hidden = F.relu(self.fc1(x))
        return F.relu(self.fc2(hidden))


class FrozenLatentDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.network(z).reshape(-1, 1, 28, 28)
