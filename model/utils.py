import os
from config import * 
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

def visualize_latent_space(model, dataloader, epoch=0, output_dir="visualizations"):
    """
    Visualizes the latent space of the VAE model using PCA.
    
    Encodes all data from the dataloader, performs PCA to reduce the latent space
    to 2D, and creates a scatter plot colored by the average pitch of each sample.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print(f"\n--- Visualising latent space for epoch: {epoch} ---")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    all_z_means = []
    all_avg_pitches = [] 

    # --- 1. Encode all data and calculate attributes ---
    with torch.no_grad():
        for pianorolls in tqdm(dataloader, desc="Encoding samples"):
            pianorolls = pianorolls.to(device)

            # Get the mean of the latent distribution from the encoder
            mean, _ = model.encoder(pianorolls)
            all_z_means.append(mean.cpu().numpy())

            # Calculate the average pitch for each piano roll in the batch
            for i in range(pianorolls.size(0)):
                # CORRECTED: Get the full piano roll, no 'lengths' needed.
                single_pianoroll = pianorolls[i].cpu().numpy()
                
                # CORRECTED: Find active notes (ATTACK or HOLD)
                note_indices = np.where(single_pianoroll > 0)
                
                # CORRECTED: Get the pitches (second element of the tuple)
                pitches = note_indices[1]
                
                if len(pitches) > 0:
                    # CORRECTED: Calculate the mean of the pitches
                    avg_pitch = np.mean(pitches)
                    all_avg_pitches.append(avg_pitch)
                else:
                    # Handle silent piano rolls
                    all_avg_pitches.append(0)

    all_z_means = np.concatenate(all_z_means, axis=0)
    all_avg_pitches = np.array(all_avg_pitches)

    # --- 2. Perform PCA to reduce dimensionality to 2D ---
    print("Performing PCA on the latent space...")
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_z_means)

    # --- 3. Create and save the plot ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        z_2d[:, 0], 
        z_2d[:, 1], 
        c=all_avg_pitches, 
        cmap='viridis', # 'viridis' is great for pitch (low=purple, high=yellow)
        alpha=0.7,
        s=15
    )

    # CORRECTED: Set the correct label for the color bar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Average Pitch (MIDI Note Number)', fontsize=12)

    ax.set_title(f'Latent Space Visualization (Epoch {epoch})', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.grid(True)

    output_filename = os.path.join(output_dir, f"latent_space_epoch_{epoch}.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Saving plot to: {output_filename}\n")
    plt.show()
