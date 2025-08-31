"""
State representation:
- 0 (OFF): Note is off (color: white).
- 1 (ATTACK): Note has started playing (color: green).
- 2 (HOLD): Note is being played (color: blue).
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import mido
import numpy as np
import os
import glob
from config import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

STATE_OFF = 0
STATE_ATTACK = 1
STATE_HOLD = 2

class MidiDataset(Dataset):
    def __init__(self, midi_dir: str, num_bars: int, steps_per_bar: int = 32, verbose: bool = False):
        self.midi_dir = midi_dir
        self.num_bars = num_bars
        self.steps_per_bar = steps_per_bar
        self.num_steps = num_bars * steps_per_bar
        self.midi_files = glob.glob(os.path.join(midi_dir, '*.mid')) + \
                          glob.glob(os.path.join(midi_dir, '*.midi'))
        self.verbose = verbose

        print(f"Dataset contains {len(self.midi_files)} MIDI samples.")
        
        self.sequences = []
        self.file_paths = []
        for file_path in self.midi_files:
            try:
                piano_roll = self._process_midi_file(file_path)
                if piano_roll is not None:
                    self.sequences.append(piano_roll)
                    self.file_paths.append(file_path)
            except Exception as e:
                print(f"Error processing file {os.path.basename(file_path)}: {e}")

        print(f"Succesfully processed {len(self.sequences)} MIDI samples.")

    def _process_midi_file(self, file_path: str) -> torch.Tensor | None:
        mid = mido.MidiFile(file_path, clip=True)
        if mid.ticks_per_beat == 0: return None
        ticks_per_beat = mid.ticks_per_beat
        ticks_per_bar = ticks_per_beat * 4
        ticks_per_step = ticks_per_bar / self.steps_per_bar
        piano_roll = np.full((self.num_steps, 128), STATE_OFF, dtype=np.int8)
        events = []
        for track in mid.tracks:
            current_time_ticks = 0
            for msg in track:
                current_time_ticks += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    events.append({'type': 'on', 'note': msg.note, 'time': current_time_ticks})
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    events.append({'type': 'off', 'note': msg.note, 'time': current_time_ticks})
        events.sort(key=lambda x: x['time'])
        active_notes = {}
        for event in events:
            step = int(event['time'] / ticks_per_step)
            if step >= self.num_steps: continue
            note = event['note']
            if event['type'] == 'on':
                piano_roll[step, note] = STATE_ATTACK
                active_notes[note] = step
            elif event['type'] == 'off' and note in active_notes:
                start_step = active_notes[note]
                for i in range(start_step + 1, step + 1):
                    if i < self.num_steps and piano_roll[i, note] == STATE_OFF:
                        piano_roll[i, note] = STATE_HOLD
                del active_notes[note]
                
        # Find the first time step with a note attack
        attack_steps = np.where(np.any(piano_roll == STATE_ATTACK, axis=1))[0]
        
        if len(attack_steps) > 0:
            first_attack_step = attack_steps[0]
            
            # If there's leading silence, shift the piano roll
            if first_attack_step > 0:
                # Create a new empty piano roll
                new_piano_roll = np.full_like(piano_roll, STATE_OFF)
                
                # Copy the sequence starting from the first attack
                trimmed_part = piano_roll[first_attack_step:]
                new_piano_roll[:len(trimmed_part)] = trimmed_part
                
                piano_roll = new_piano_roll
        else:
            # If no notes were found in the sequence, return None to skip this file
            return None
            
        return torch.tensor(piano_roll, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        if self.verbose:
            return self.sequences[idx], self.file_paths[idx]
        else:
            return self.sequences[idx]

    @staticmethod
    def visualize(piano_roll_tensor: torch.Tensor, title: str = "Piano roll Visualization"):
        piano_roll = piano_roll_tensor.cpu().numpy().T
        cmap = ListedColormap(["#000000", "#F03528", "#EDF030"])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(piano_roll, aspect='auto', cmap=cmap, interpolation='nearest', origin='lower')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Krok czasowy", fontsize=12)
        ax.set_ylabel("Nuta MIDI", fontsize=12)
        y_ticks = np.arange(0, 128, 12)
        y_labels = [f"C{i-1}" for i in range(len(y_ticks))]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(20, 100)
        legend_patches = [
            mpatches.Patch(color="#F03528", label='Atack'),
            mpatches.Patch(color="#EDF030", label='Hold')
        ]
        ax.legend(handles=legend_patches, loc='upper right')
        plt.grid(True, which='both', axis='x', linestyle=':', color='grey', alpha=0.5)
        plt.tight_layout()
        plt.show()

def prepare_dataloaders(dataset_dir, split_ratios = (85,15), seed: int = 42):
        torch.manual_seed(seed)

        print("Loading dataset...")
        dataset = MidiDataset(
            midi_dir=DATASET_DIR,
            num_bars=NUM_BARS,
            steps_per_bar=STEPS_PER_BAR
        )

        total_size = len(dataset)
        train_size = int(split_ratios[0] * total_size)
        val_size = total_size - train_size

        # Podział datasetu na podzbiory
        print("Splitting dataset...")
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print(f"Dataset split into: Train={len(train_dataset)}, Validation={len(val_dataset)}")

        # Utworzenie DataLoaderów
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        print("Finished preparing dataset.\n")
        return train_dataloader, val_dataloader