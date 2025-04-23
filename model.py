from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class config:
    def __init__(self, hidden_dims, input_dim):
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim


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

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        mean, log_var = torch.split(x, split_size_or_sections=[self.z, self.z], dim=-1)
        return mean, log_var


def reparameterize(mean, log_var, num_sampler_per_z=1):
    batch_size,latent_dim=mean.shape
    std = torch.exp(0.5 * log_var)
    eps = torch.rand_like(std)
    z = mean + std * eps
    return z


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
