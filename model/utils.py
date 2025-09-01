import os
from config import * 
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import mido
STATE_OFF = 0
STATE_ATTACK = 1
STATE_HOLD = 2

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

def calculate_class_weights(dataloader):
    """
    Calculates inverse frequency weights for the 3 classes (OFF, ATTACK, HOLD).
    """
    print("Calculating class weights...")
    # Initialize counts for 3 classes
    class_counts = torch.zeros(3)
    
    for batch in tqdm(dataloader, desc="Analyzing dataset for weights"):
        # Flatten the batch to a 1D tensor of class labels
        labels_flat = batch.view(-1)
        
        # Count occurrences of each class (0, 1, 2)
        class_counts += torch.bincount(labels_flat, minlength=3)
            
    # Calculate inverse frequency
    total_counts = class_counts.sum()
    class_weights = total_counts / class_counts
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum()
    
    print(f"Calculated class weights: {class_weights.tolist()}")
    return class_weights

# Dodaj tę funkcję i potrzebne importy do swojego pliku utils.py



def tensor_to_midi(piano_roll_tensor: torch.Tensor, output_path: str, 
                   ticks_per_beat: int = 480, tempo_bpm: int = 120):
    """
    Converts a 3-state piano roll tensor into a MIDI file.

    Args:
        piano_roll_tensor (torch.Tensor): The (num_steps, 128) tensor with states 0, 1, 2.
        output_path (str): Path to save the output .mid file.
        ticks_per_beat (int): The MIDI file's time resolution.
        tempo_bpm (int): The tempo of the resulting piece in beats per minute.
    """
    from config import STEPS_PER_BAR # Import locally to get time context

    print(f"Converting tensor to MIDI file at {output_path}...")
    
    # Ensure tensor is on CPU and is a numpy array
    piano_roll = piano_roll_tensor.cpu().numpy()
    num_steps, num_notes = piano_roll.shape
    
    # Calculate how many MIDI ticks correspond to one step in our piano roll
    ticks_per_step = (ticks_per_beat * 4) / STEPS_PER_BAR # Assuming 4/4 time

    # --- Create MIDI File and Track ---
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo (required for correct playback speed)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm)))
    # Set instrument to Acoustic Grand Piano (program 0)
    track.append(mido.Message('program_change', program=0, time=0))

    # --- Find note start and end events ---
    events = []
    for pitch in range(num_notes):
        note_start_step = -1
        for step in range(num_steps):
            state = piano_roll[step, pitch]
            
            # If a note starts (state is ATTACK)
            if state == STATE_ATTACK and note_start_step < 0:
                note_start_step = step
            
            # If a note ends (state is OFF after being ON)
            elif state == STATE_OFF and note_start_step >= 0:
                duration_steps = step - note_start_step
                # Add note_on and note_off events with absolute time in steps
                events.append({'type': 'note_on', 'pitch': pitch, 'step': note_start_step})
                events.append({'type': 'note_off', 'pitch': pitch, 'step': note_start_step + duration_steps})
                note_start_step = -1
        
        # Handle notes that are still on at the very end of the sequence
        if note_start_step >= 0:
            duration_steps = num_steps - note_start_step
            events.append({'type': 'note_on', 'pitch': pitch, 'step': note_start_step})
            events.append({'type': 'note_off', 'pitch': pitch, 'step': note_start_step + duration_steps})

    # --- Convert events to MIDI messages with relative time ---
    if not events:
        print("Warning: No notes found in the tensor. Saving an empty MIDI file.")
        mid.save(output_path)
        return

    # Sort events chronologically by step
    events.sort(key=lambda e: e['step'])
    
    last_event_ticks = 0
    for event in events:
        current_event_ticks = int(event['step'] * ticks_per_step)
        delta_ticks = current_event_ticks - last_event_ticks
        
        track.append(mido.Message(
            event['type'],
            note=event['pitch'],
            velocity=80, # Use a standard velocity for generated notes
            time=delta_ticks
        ))
        last_event_ticks = current_event_ticks

    # Save the final MIDI file
    mid.save(output_path)
    print("MIDI file saved successfully.")