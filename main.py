# main.py
import argparse
import torch
import matplotlib.pyplot as plt
from train import train_model
from diffusion import GaussianDiffusion1D

def plot_samples(samples, rows, cols):
    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(samples[i, 0], label='Channel 1')
        ax.plot(samples[i, 1], label='Channel 2')
        ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("Result/Diffusion.pdf")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train a DDPM model for time series generation')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the labels file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to train on (e.g., "cuda:1")')
    parser.add_argument('--timesteps', type=int, default=500, help='Number of diffusion timesteps')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for training')
    args = parser.parse_args()

    model, gaussian_diffusion = train_model(
        args.data_path,
        args.labels_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        timesteps=args.timesteps,
        learning_rate=args.learning_rate
    )

    length = 1024
    batch_size = 16
    samples = gaussian_diffusion.sample(model, length, batch_size=batch_size, channels=2)

    samples = samples.cpu().numpy()
    plot_samples(samples, rows=4, cols=4)

if __name__ == '__main__':
    main()
