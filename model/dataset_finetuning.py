# dataset.py

"""
State representation:
- 0 (OFF): Note is off.
- 1 (ATTACK): Note has started playing.
- 2 (HOLD): Note is being played.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import mido
import numpy as np
import os
import glob
from finetune_config import *
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# --- State Constants ---
STATE_OFF = 0
STATE_ATTACK = 1
STATE_HOLD = 2

class MidiDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing MIDI files into a 3-state piano roll format.
    
    This version processes all tracks from a MIDI file into a single piano roll and
    supports sliding window extraction.
    """
    def __init__(self, midi_dir: str, num_bars: int, steps_per_bar: int = 16, 
                 use_sliding_window: bool = True, stride_in_bars: int = 1, verbose: bool = False):
        self.midi_dir = midi_dir
        self.num_bars = num_bars
        self.steps_per_bar = steps_per_bar
        self.num_steps_per_segment = num_bars * steps_per_bar
        self.use_sliding_window = use_sliding_window
        self.stride_in_steps = stride_in_bars * steps_per_bar
        self.verbose = verbose

        self.midi_files = glob.glob(os.path.join(midi_dir, '*.mid')) + \
                          glob.glob(os.path.join(midi_dir, '*.midi'))

        if not self.midi_files:
            raise FileNotFoundError(f"No .mid/.midi files found in directory: {midi_dir}")

        print(f"Found {len(self.midi_files)} MIDI files.")
        
        self.sequences = []
        self.file_paths = []
        
        print("Processing MIDI files and extracting segments...")
        for file_path in tqdm(self.midi_files, desc="Processing files"):
            try:
                full_piano_roll = self._process_full_midi_file(file_path)
                
                if full_piano_roll is None or full_piano_roll.shape[0] < self.num_steps_per_segment:
                    continue

                if self.use_sliding_window:
                    total_steps = full_piano_roll.shape[0]
                    for start_step in range(0, total_steps - self.num_steps_per_segment + 1, self.stride_in_steps):
                        end_step = start_step + self.num_steps_per_segment
                        segment = full_piano_roll[start_step:end_step]
                        
                        trimmed_segment = self._trim_leading_silence(segment)
                        if trimmed_segment is not None:
                            self.sequences.append(torch.tensor(trimmed_segment, dtype=torch.long))
                            self.file_paths.append(file_path)
                else:
                    segment = full_piano_roll[:self.num_steps_per_segment]
                    trimmed_segment = self._trim_leading_silence(segment)
                    if trimmed_segment is not None:
                        self.sequences.append(torch.tensor(trimmed_segment, dtype=torch.long))
                        self.file_paths.append(file_path)

            except Exception as e:
                print(f"Warning: Error processing file {os.path.basename(file_path)}: {e}")

        if not self.sequences:
            raise ValueError("Failed to extract any valid sequences from the provided MIDI files.")

        print(f"Successfully extracted {len(self.sequences)} segments from {len(self.midi_files)} files.")

    def _trim_leading_silence(self, piano_roll: np.ndarray) -> np.ndarray | None:
        """Helper function to remove leading silence from a piano roll segment."""
        attack_steps = np.where(np.any(piano_roll == STATE_ATTACK, axis=1))[0]
        if len(attack_steps) > 0:
            first_attack_step = attack_steps[0]
            if first_attack_step > 0:
                new_piano_roll = np.full_like(piano_roll, STATE_OFF)
                trimmed_part = piano_roll[first_attack_step:]
                new_piano_roll[:len(trimmed_part)] = trimmed_part
                return new_piano_roll
            return piano_roll
        return None 

    def _process_full_midi_file(self, file_path: str) -> np.ndarray | None:
        """Processes an entire MIDI file into a single long 3-state piano roll."""
        mid = mido.MidiFile(file_path, clip=True)
        if mid.ticks_per_beat == 0: return None
        
        ticks_per_beat = mid.ticks_per_beat
        ticks_per_bar = ticks_per_beat * 4
        ticks_per_step = ticks_per_bar / self.steps_per_bar
        
        events = []
        total_time_ticks = 0
        # Process ALL tracks in the file
        for track in mid.tracks:
            current_time_ticks = 0
            for msg in track:
                current_time_ticks += msg.time
                total_time_ticks = max(total_time_ticks, current_time_ticks)
                if msg.type == 'note_on' and msg.velocity > 0:
                    events.append({'type': 'on', 'note': msg.note, 'time': current_time_ticks})
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    events.append({'type': 'off', 'note': msg.note, 'time': current_time_ticks})
        
        if not events: return None
        events.sort(key=lambda e: e['time'])
        
        total_steps = int(total_time_ticks / ticks_per_step) + 1
        piano_roll = np.full((total_steps, 128), STATE_OFF, dtype=np.int8)
        
        active_notes = {}
        for event in events:
            step = int(event['time'] / ticks_per_step)
            if step >= total_steps: continue
            note = event['note']
            if event['type'] == 'on':
                piano_roll[step, note] = STATE_ATTACK
                active_notes[note] = step
            elif event['type'] == 'off' and note in active_notes:
                start_step = active_notes[note]
                for i in range(start_step + 1, step + 1):
                    if i < total_steps and piano_roll[i, note] == STATE_OFF:
                        piano_roll[i, note] = STATE_HOLD
                del active_notes[note]
        
        return piano_roll

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        # Returns a single tensor, not a tuple with BPM
        if self.verbose:
            return self.sequences[idx], self.file_paths[idx]
        else:
            return self.sequences[idx]

    @staticmethod
    def visualize(piano_roll_tensor: torch.Tensor, title: str = "Piano roll Visualization", 
                  show_plot: bool = True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))
        else:
            fig = ax.get_figure()
        piano_roll = piano_roll_tensor.cpu().numpy().T
        cmap = ListedColormap(["#000000", "#F03528", "#EDF030"])
        ax.imshow(piano_roll, aspect='auto', cmap=cmap, interpolation='nearest', origin='lower')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("MIDI Note", fontsize=12)
        y_ticks = np.arange(0, 128, 12)
        y_labels = [f"C{i-1}" for i in range(len(y_ticks))]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(20, 100)
        legend_patches = [
            mpatches.Patch(color="#F03528", label='Attack'),
            mpatches.Patch(color="#EDF030", label='Hold')
        ]
        ax.legend(handles=legend_patches, loc='upper right')
        ax.grid(True, which='both', axis='x', linestyle=':', color='grey', alpha=0.5)
        if show_plot:
            plt.tight_layout()
            plt.show()
        return fig

def prepare_dataloaders(split_ratios=(0.85, 0.15), seed: int = 42):
    """
    Simplified utility function to create and split the MidiDataset into DataLoaders.
    """
    torch.manual_seed(seed)
    
    print("Loading dataset...")
    dataset = MidiDataset(
        midi_dir=DATASET_DIR,
        num_bars=NUM_BARS,
        steps_per_bar=STEPS_PER_BAR,
        use_sliding_window=USE_SLIDING_WINDOW,
        stride_in_bars=STRIDE_IN_BARS
    )

    total_size = len(dataset)
    if total_size == 0:
        raise ValueError("Dataset is empty. Check paths or processing logic.")

    train_size = int(split_ratios[0] * total_size)
    val_size = total_size - train_size
    
    print("Splitting dataset...")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Dataset split into: Train={len(train_dataset)}, Validation={len(val_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    print("Finished preparing dataset.\n")
    return train_dataloader, val_dataloader