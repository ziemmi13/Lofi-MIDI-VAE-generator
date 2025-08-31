from config import *
import numpy as np
import pretty_midi
from config import * # Import all your configuration variables
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def print_config(config):
    print("--- Data Configuration ---")
    for key in dir(config.data):
        if not key.startswith('__'):
            value = getattr(config.data, key)
            print(f"{key}: {value}")

    print("\n--- Model Configuration ---")
    for key in dir(config.model):
        if not key.startswith('__'):
            value = getattr(config.model, key)
            if not isinstance(value, type):
                print(f"{key}: {value}")

    print("\n--- Training Configuration ---")
    for key in dir(config.train):
        if not key.startswith('__'):
            value = getattr(config.train, key)
            print(f"{key}: {value}")

def visualize_latent_space(model, dataloader, output_filename="latent_space.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print("\n--- Visualising latent space ---")
    all_z_means = []
    all_avg_pitches = [] 

    with torch.no_grad():
        for pianorolls, lengths in dataloader:
            pianorolls = pianorolls.to(device)

            mean, _ = model.encoder(pianorolls, lengths)
            
            all_z_means.append(mean.cpu().numpy())

            for i in range(pianorolls.size(0)):
                single_pianoroll = pianorolls[i, :, :lengths[i]].cpu().numpy()
                note_indices = np.where(single_pianoroll > 0.1)
                if len(note_indices[0]) > 0:
                    avg_pitch = np.mean(note_indices[0])
                    all_avg_pitches.append(avg_pitch)
                else:
                    all_avg_pitches.append(0)

    all_z_means = np.concatenate(all_z_means, axis=0)
    all_avg_pitches = np.array(all_avg_pitches)

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_z_means)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        z_2d[:, 0], 
        z_2d[:, 1], 
        c=all_avg_pitches, 
        cmap='viridis',
        alpha=0.7,
        s=15
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Average velocity')

    ax.set_title('Latent Space visualisation', fontsize=16)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.grid(True)

    plt.savefig(output_filename, dpi=300)
    print(f"Saving plot to: {output_filename}\n")
    plt.show()


