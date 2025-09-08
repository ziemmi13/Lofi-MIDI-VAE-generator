import os
from config import * 
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import mido
from dataset import MidiDataset, prepare_dataloaders

# --- State Constants ---
STATE_OFF = 0
STATE_ATTACK = 1
STATE_HOLD = 2

def visualize_latent_space(model, dataloader, output_dir="visualizations"):
    """
    Visualizes the latent space of the VAE model using PCA.
    
    Encodes all data from the dataloader, performs PCA to reduce the latent space
    to 2D, and creates a scatter plot colored by the average pitch of each sample.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print(f"\n--- Visualising latent space")
    os.makedirs(output_dir, exist_ok=True)
    
    all_z_means = []
    all_avg_pitches = [] 

    with torch.no_grad():
        # ZMIANA: Prawidłowe rozpakowanie danych z dataloadera
        for pianorolls, _ in tqdm(dataloader, desc="Encoding samples"):
            pianorolls = pianorolls.to(device)

            mean, _ = model.encoder(pianorolls)
            all_z_means.append(mean.cpu().numpy())

            for i in range(pianorolls.size(0)):
                single_pianoroll = pianorolls[i].cpu().numpy()
                note_indices = np.where(single_pianoroll > 0)
                pitches = note_indices[1]
                
                if len(pitches) > 0:
                    avg_pitch = np.mean(pitches)
                    all_avg_pitches.append(avg_pitch)
                else:
                    all_avg_pitches.append(0)

    all_z_means = np.concatenate(all_z_means, axis=0)
    all_avg_pitches = np.array(all_avg_pitches)

    print("Performing PCA on the latent space...")
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
    cbar.set_label('Average Pitch (MIDI Note Number)', fontsize=12)

    ax.set_title(f'Latent Space Visualization (Epoch {epoch})', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.grid(True)

    output_filename = os.path.join(output_dir, f"latent_space_epoch_{epoch}.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Saving plot to: {output_filename}\n")
    plt.show(block=False) 

    return pca, z_2d

def calculate_class_weights(dataloader):
    """
    Calculates inverse frequency weights for the 3 classes (OFF, ATTACK, HOLD).
    """
    print("Calculating class weights...")
    class_counts = torch.zeros(3)
    
    for pianorolls, _ in tqdm(dataloader, desc="Analyzing dataset for weights"):
        labels_flat = pianorolls.view(-1)
        class_counts += torch.bincount(labels_flat, minlength=3)
            
    total_counts = class_counts.sum()
    # Dodaj małą wartość epsilon, aby uniknąć dzielenia przez zero, jeśli klasa nie występuje
    class_weights = total_counts / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum()
    
    print(f"Calculated class weights: {class_weights.tolist()}")
    return class_weights

def tensor_to_midi(piano_roll_tensor: torch.Tensor, output_path: str, 
                   bpm: int, ticks_per_beat: int = 16):
    """
    Converts a 3-state piano roll tensor into a MIDI file.
    This corrected version properly handles note durations and re-articulations.

    Args:
        piano_roll_tensor (torch.Tensor): The (num_steps, 128) tensor with states 0, 1, 2.
        output_path (str): Path to save the output .mid file.
        ticks_per_beat (int): The MIDI file's time resolution.
        tempo_bpm (int): The tempo of the resulting piece in beats per minute.
    """
    from config import STEPS_PER_BAR

    print(f"Converting tensor to MIDI file at {output_path}...")
    
    piano_roll = piano_roll_tensor.cpu().numpy()
    num_steps, num_notes = piano_roll.shape
    
    ticks_per_step = (ticks_per_beat * 4) / STEPS_PER_BAR

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    track.append(mido.Message('program_change', program=0, time=0))

    events = []
    for pitch in range(num_notes):
        for step in range(num_steps):
            if piano_roll[step, pitch] == STATE_ATTACK:
                note_on_step = step
                note_off_step = note_on_step + 1 # Nuta musi trwać co najmniej jeden krok

                for end_step in range(note_on_step + 1, num_steps):
                    if piano_roll[end_step, pitch] == STATE_OFF:
                        note_off_step = end_step
                        break
                    note_off_step = num_steps 
                
                events.append({'type': 'note_on', 'pitch': pitch, 'step': note_on_step})
                events.append({'type': 'note_off', 'pitch': pitch, 'step': note_off_step})

    if not events:
        print("Warning: No notes found in the tensor. Saving an empty MIDI file.")
        mid.save(output_path)
        return

    events.sort(key=lambda e: e['step'])
    
    last_event_ticks = 0
    for event in events:
        current_event_ticks = int(event['step'] * ticks_per_step)
        delta_ticks = current_event_ticks - last_event_ticks
        
        # Użyj velocity=0 dla note_off, co jest standardową praktyką
        velocity = 80 if event['type'] == 'note_on' else 0
        
        track.append(mido.Message(
            event['type'],
            note=event['pitch'],
            velocity=velocity,
            time=delta_ticks
        ))
        last_event_ticks = current_event_ticks

    # Upewnij się, że katalog docelowy istnieje
    mid.save(output_path)
    print("MIDI file saved successfully.")

def explore_latent_space(model, data_loader, points_to_explore):
    model.eval()
    pca, z_2d = visualize_latent_space(model, data_loader)

    # --- 3. Interactive Exploration Loop ---
    print("\n--- Latent Space Explorer ---")
            
    for x, y in points_to_explore:
        print(f"Generating sample from point ({x}, {y})...")
        # --- 4. Inverse Transform and Generation ---
        with torch.no_grad():

            # Create a 2D point and reshape it for the inverse transform
            point_2d = np.array([[x, y]])
            
            # Use the fitted PCA object to transform the 2D point back to the high-dimensional space
            point_high_dim = pca.inverse_transform(point_2d)
            
            # Convert to a PyTorch tensor and move to the correct device
            z_vector = torch.from_numpy(point_high_dim).float().to(model.device)
            
            # Generate a piano roll using only the decoder
            generated_pianoroll = model.decoder(z_vector)
            final_pianoroll = torch.argmax(generated_pianoroll, dim=-1).squeeze(0)
        
        # Visualize the generated piano roll
        MidiDataset.visualize(final_pianoroll.cpu(), title=f"Generated from ({x:.2f}, {y:.2f})")
        
        # Save the generated sample to a MIDI file
        output_dir = "generated_explorations"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"sample_x{x:.2f}_y{y:.2f}.mid")
        tensor_to_midi(final_pianoroll, output_path, bpm=100)
    