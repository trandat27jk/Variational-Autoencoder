import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import VAE
from model import config as Config
from model import loss as vae_loss

# Config
cfg = Config(hidden_dims=[128, 64, 36, 18, 18], input_dim=784, samples_per_z=6)
epoch = 50
use_amp = True
batch_size = 32
num_workers = int(os.cpu_count() // 2)
learning_rate = 0.001
grad_norm = 1.0
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VAE(cfg).to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

# Training Loop
for epoch_idx in range(epoch):
    for source, _ in train_loader:
        source = source.to(device)
        source = source.reshape(source.size(0), -1)

        with torch.cuda.amp.autocast(dtype=dtype, enabled=use_amp):
            output = model(source)
            loss_val = vae_loss(
                mean=output["mean"],
                varlog=output["log_var"],
                predict=output["imgs"],
                target=source,
            )

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss_val).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm)
            optimizer.step()
        torch.save(model.state_dict(), "vae_model.pth")
        print(f"Epoch [{epoch_idx+1}/{epoch}], Loss: {loss_val.item():.4f}")
