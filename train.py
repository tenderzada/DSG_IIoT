# train.py
import torch
from torch.optim import Adam
from diffusion import GaussianDiffusion1D
from model import UNetModel1D
from data import load_data

def train_model(data_path, labels_path, batch_size=64, num_epochs=10, device='cuda:1', timesteps=500, learning_rate=5e-4):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_loader, _ = load_data(data_path, labels_path, batch_size=batch_size)

    model = UNetModel1D(
        in_channels=2,
        model_channels=128,
        out_channels=2,
        num_res_blocks=2,
        dropout=0.1
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    gaussian_diffusion = GaussianDiffusion1D(timesteps=timesteps)

    for epoch in range(num_epochs):
        for step, (batch, _) in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            t = torch.randint(0, timesteps, (batch.shape[0],), device=device).long()
            loss = gaussian_diffusion.train_losses(model, batch, t)
            if step % 100 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
    return model, gaussian_diffusion
