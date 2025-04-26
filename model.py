from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class config:
    def __init__(self, hidden_dims, input_dim, samples_per_z):
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.samples_per_z = samples_per_z


# encoder
class encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.z = config.hidden_dims[-1] // 2
        assert (
            config.hidden_dims[-1] % 2 == 0
        ), "Last hidden layer must be divisible by 2 to split into mean/logvar"
        prev_dim = config.input_dim
        for i, dim in enumerate(config.hidden_dims):
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
            if i < len(config.hidden_dims) - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        mean, log_var = torch.split(x, split_size_or_sections=[self.z, self.z], dim=-1)
        return mean, log_var


class decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        hidden_dims = list(reversed(config.hidden_dims))
        hidden_dims.append(config.input_dim)
        prev_dim = hidden_dims[0] // 2
        for i, dim in enumerate(hidden_dims[1:]):
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
            if i < len(hidden_dims) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def reparameterize(mean, log_var, sample_per_z=1):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mean + std * eps
    return z


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = encoder(config)
        self.decoder = decoder(config)
        self.config = config

    def forward(self, x):
        mean, logvar = self.encoder(x)
        batch_size, latent_dim = mean.shape

        if self.config.samples_per_z > 1:
            mean = mean.unsqueeze(1).expand(
                batch_size, self.config.samples_per_z, latent_dim
            )
            logvar = logvar.unsqueeze(1).expand(
                batch_size, self.config.samples_per_z, latent_dim
            )

            mean = mean.contiguous().view(
                batch_size * self.config.samples_per_z, latent_dim
            )
            logvar = logvar.contiguous().view(
                batch_size * self.config.samples_per_z, latent_dim
            )

        z = reparameterize(mean, logvar)
        x_probs = self.decoder(z)
        x_probs = x_probs.reshape(batch_size, self.config.samples_per_z, -1)
        x_probs = torch.mean(x_probs, dim=1)

        return {"imgs": x_probs, "z": z, "mean": mean, "log_var": logvar}

    def inference(self, num_samples=1):
        z = torch.randn(num_samples, self.encoder.z)
        output = self.decoder(z)
        return output


# loss function
def loss(mean, varlog, predict, target):
    kl_loss = -0.5 * torch.sum(1 + varlog - mean.pow(2) - varlog.exp(), dim=1)
    kl_loss = kl_loss.mean()
    # recon_loss
    recon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        predict, target, reduction="mean"
    )
    loss = kl_loss + recon_loss
    return loss
