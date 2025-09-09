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

    ax.set_title(f'Latent Space Visualization', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.grid(True)

    output_filename = os.path.join(output_dir, f"latent_space.png")
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

# You should have these constants defined, matching your MidiDataset
MIN_PITCH = 36  # C2
STATE_ATTACK = 1
STATE_HOLD = 2
STATE_OFF = 0

def tensor_to_midi(piano_roll_tensor: torch.Tensor, output_path: str,
                   bpm: int, ticks_per_beat: int = 480, min_pitch: int = MIN_PITCH):
    """
    Converts a 3-state piano roll tensor (with a limited pitch range) into a MIDI file.
    This corrected version properly handles note durations, re-articulations, and maps
    the pitch index back to the correct MIDI note number.

    Args:
        piano_roll_tensor (torch.Tensor): The (num_steps, num_pitches) tensor with states 0, 1, 2.
        output_path (str): Path to save the output .mid file.
        bpm (int): The tempo of the resulting piece in beats per minute.
        ticks_per_beat (int): The MIDI file's time resolution. A standard value is 480.
        min_pitch (int): The lowest MIDI note number represented by index 0 in the piano roll.
    """
    # Assuming STEPS_PER_BAR is defined elsewhere, e.g., from a config file
    # from config import STEPS_PER_BAR
    STEPS_PER_BAR = 16 # Or import it

    print(f"Converting tensor to MIDI file at {output_path}...")
    
    piano_roll = piano_roll_tensor.cpu().numpy()
    num_steps, num_pitches = piano_roll.shape
    
    # Calculate how many MIDI ticks one step in the piano roll represents
    # A bar has 4 beats. ticks_per_bar = ticks_per_beat * 4
    ticks_per_step = (ticks_per_beat * 4) / STEPS_PER_BAR

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    track.append(mido.Message('program_change', program=0, time=0))

    events = []
    for pitch_idx in range(num_pitches):
        for step in range(num_steps):
            # A new note starts on an ATTACK state
            if piano_roll[step, pitch_idx] == STATE_ATTACK:
                note_on_step = step
                note_off_step = note_on_step + 1 # Default duration of 1 step

                # Find the end of the note by scanning forward
                for end_step in range(note_on_step + 1, num_steps):
                    # The note ends if the state is no longer HOLD
                    if piano_roll[end_step, pitch_idx] != STATE_HOLD:
                        note_off_step = end_step
                        break
                else: # If the loop completes, the note holds until the end
                    note_off_step = num_steps
                
                # The actual MIDI note is the index + the minimum pitch
                midi_note = pitch_idx + min_pitch
                
                events.append({'type': 'note_on', 'pitch': midi_note, 'step': note_on_step, 'velocity': 80})
                events.append({'type': 'note_off', 'pitch': midi_note, 'step': note_off_step, 'velocity': 0})

    if not events:
        print("Warning: No notes found in the tensor. Saving an empty MIDI file.")
        mid.save(output_path)
        return

    # Sort all events by their step time
    events.sort(key=lambda e: e['step'])
    
    last_event_ticks = 0
    for event in events:
        current_event_ticks = int(event['step'] * ticks_per_step)
        delta_ticks = current_event_ticks - last_event_ticks
        
        track.append(mido.Message(
            event['type'],
            note=event['pitch'],
            velocity=event['velocity'],
            time=delta_ticks
        ))
        last_event_ticks = current_event_ticks
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mid.save(output_path)
    print("MIDI file saved successfully.")