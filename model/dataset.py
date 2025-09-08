import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import mido
import numpy as np
import os
from config import * 
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

STATE_OFF = 0
STATE_ATTACK = 1
STATE_HOLD = 2

DRUMS_INDEX = 1
BASS_INDEX = 2
PIANO_INDEX = 3
# Index 0 is metadata

class MidiDataset(Dataset):

    def __init__(self, midi_dir: str, csv_path: str, num_bars: int, steps_per_bar: int,
                 use_sliding_window: bool, stride_in_bars: int, verbose: bool = False):
        self.midi_dir = midi_dir
        self.num_bars = num_bars
        self.steps_per_bar = steps_per_bar
        self.num_steps_per_segment = num_bars * steps_per_bar
        self.use_sliding_window = use_sliding_window
        self.stride_in_steps = stride_in_bars * steps_per_bar
        self.verbose = verbose

        print(f"Loading metadata from {csv_path}...")
        self.df = pd.read_csv(csv_path)

        
        self.piano_rolls = []
        self.bpms = []
        self.file_paths = []
        print("Processing MIDI files...")
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing files"):
            try:
                file_name = row['filename']
                file_path = os.path.join(self.midi_dir, file_name)
                bpm = round(row['tempo'])
                first_beat_seconds = float(row['first_beat_time'])
                
                # Get pianoroll
                piano_roll, ticks_per_beat = self.midi_to_pianoroll(file_path)

                # --- Bar-aware Slicing Logic ---
                ticks_per_step = (ticks_per_beat * 4) / self.steps_per_bar
                ticks_per_second = ticks_per_beat * (bpm / 60.0)
                first_beat_ticks = first_beat_seconds * ticks_per_second
                start_offset_in_steps = int(round(first_beat_ticks / ticks_per_step))

                if self.use_sliding_window:
                    total_steps = piano_roll.shape[0]
                    for start in range(start_offset_in_steps, total_steps - self.num_steps_per_segment + 1, self.stride_in_steps):
                        end = start + self.num_steps_per_segment
                        segment = piano_roll[start:end]
                        if np.any(segment == STATE_ATTACK):
                            self.piano_rolls.append(torch.tensor(segment, dtype=torch.long))
                            self.bpms.append(bpm)
                            self.file_paths.append(file_path)

            except Exception as e:
                print(f"Warning: Error processing file {row['filename']}: {e}")

        if not self.piano_rolls:
            raise ValueError("Failed to process any valid MIDI files from the CSV list.")
        
        print(f"Successfully extracted {len(self.piano_rolls)} bar-aligned segments.")

    def midi_to_pianoroll(self, file_path: str) -> tuple[np.ndarray, int] | tuple[None, None]:
        """Converts a MIDI file to a 3-state piano roll and returns it with its ticks_per_beat."""
        mid = mido.MidiFile(file_path, clip=True)
        
        ticks_per_beat = mid.ticks_per_beat
        ticks_per_step = (ticks_per_beat * 4) / self.steps_per_bar
        
        piano_track = mid.tracks[PIANO_INDEX]

        events, total_time_ticks = [], 0
        current_time_ticks = 0
        for msg in piano_track:
            current_time_ticks += msg.time
            total_time_ticks = max(total_time_ticks, current_time_ticks)
            if msg.type in ('note_on', 'note_off'):
                events.append({'type': 'on' if msg.type == 'note_on' and msg.velocity > 0 else 'off', 
                                'note': msg.note, 'time': current_time_ticks})
        
        if not events: return None, None
        events.sort(key=lambda e: e['time'])
        
        total_steps = int(total_time_ticks / ticks_per_step) + 1
        # Initialize an empty pianoroll
        piano_roll = np.full((total_steps, 128), STATE_OFF, dtype=np.int8)
        
        # Fill the pianoroll with active notes
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
        
        return piano_roll, ticks_per_beat
    
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
        y_labels = [f"C{i-2}" for i in range(len(y_ticks))]
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
    
    def __len__(self) -> int:
        return len(self.piano_rolls)

    def __getitem__(self, idx: int):
        if self.verbose:
            return self.piano_rolls[idx], self.bpms[idx], self.file_paths[idx]
        else:
            return self.piano_rolls[idx], self.bpms

def prepare_dataloaders(csv_path: str, split_ratios=(0.85, 0.15), seed: int = 42):
    torch.manual_seed(seed)
    print("Loading and preparing dataset...")
    dataset = MidiDataset(
        midi_dir=DATASET_DIR,
        csv_path=csv_path,
        num_bars=NUM_BARS,
        steps_per_bar=STEPS_PER_BAR,
        use_sliding_window=USE_SLIDING_WINDOW,
        stride_in_bars=STRIDE_IN_BARS
    )

    total_size = len(dataset)
    if total_size == 0:
        raise ValueError("Dataset is empty after processing.")

    train_size = int(split_ratios[0] * total_size)
    val_size = total_size - train_size
    
    print("Splitting dataset...")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Dataset split into: Train={len(train_dataset)}, Validation={len(val_dataset)}")
    
    # Use the custom collate_fn in the DataLoader
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