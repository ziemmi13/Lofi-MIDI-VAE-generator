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
from config import *
from tqdm import tqdm
import pandas as pd
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# --- State Constants ---
STATE_OFF = 0
STATE_ATTACK = 1
STATE_HOLD = 2

# --- Default value if BPM is not found ---
DEFAULT_BPM = 120.0

class MidiDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing MIDI files into a 3-state piano roll format.
    
    Features:
    - Tempo normalization to a target BPM.
    - Sliding window extraction for creating numerous samples.
    - Leading silence trimming for each sample.
    """
    def __init__(self, midi_dir: str, metadata_csv_path: str, num_bars: int, steps_per_bar: int = 16, 
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
        
        try:
            df_temp = pd.read_csv(metadata_csv_path)
            if 'bpm' not in df_temp.columns and 'tempo' in df_temp.columns:
                df_temp = df_temp.rename(columns={'tempo': 'bpm'})
            self.metadata_df = df_temp
        except FileNotFoundError:
            print(f"Warning: Metadata CSV file not found at {metadata_csv_path}. Will rely on tempo info from MIDI files.")
            self.metadata_df = None

        print(f"Found {len(self.midi_files)} MIDI files.")
        
        self.sequences = []
        self.file_paths = []
        self.bpms = []

        print("Processing MIDI files, normalizing tempo, and extracting segments...")
        for file_path in tqdm(self.midi_files, desc="Processing files"):
            try:
                file_name = os.path.basename(file_path)
                
                result = self._process_full_midi_file(file_path, file_name)
                
                if result is None:
                    continue
                
                full_piano_roll, original_bpm = result

                if full_piano_roll.shape[0] < self.num_steps_per_segment:
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
                            self.bpms.append(original_bpm)
                else:
                    segment = full_piano_roll[:self.num_steps_per_segment]
                    trimmed_segment = self._trim_leading_silence(segment)
                    if trimmed_segment is not None:
                        self.sequences.append(torch.tensor(trimmed_segment, dtype=torch.long))
                        self.file_paths.append(file_path)
                        self.bpms.append(original_bpm)

            except Exception as e:
                print(f"Warning: Error processing file {os.path.basename(file_path)}: {e}")

        if not self.sequences:
            raise ValueError("Failed to extract any valid sequences from the provided MIDI files.")

        print(f"Successfully extracted {len(self.sequences)} segments from {len(self.midi_files)} files.")

    def _trim_leading_silence(self, piano_roll: np.ndarray) -> Optional[np.ndarray]:
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

    def _get_bpm(self, file_name: str, mid: mido.MidiFile) -> float:
        """
        Gets BPM for a MIDI file. It first tries the metadata CSV, then falls back
        to a default value if not found.
        """
        if self.metadata_df is not None:
            row = self.metadata_df[self.metadata_df['filename'] == file_name]
            if not row.empty and 'bpm' in row.columns:
                bpm_val = row['bpm'].iloc[0]
                if pd.notna(bpm_val):
                    return float(bpm_val)

        # Fallback to default BPM if not in CSV
        # This part could be extended to read from MIDI meta messages if needed
        print(f"Warning: Could not find BPM for {file_name} in CSV. Using default {DEFAULT_BPM} BPM.")
        return DEFAULT_BPM

    def _process_full_midi_file(self, file_path: str, file_name: str) -> Optional[Tuple[np.ndarray, float]]:
        """
        Processes an entire MIDI file, normalizes its tempo, and returns a 3-state piano roll
        along with its original BPM.
        
        MODIFIED: This version now only processes the 3rd track of the MIDI file.
        """
        mid = mido.MidiFile(file_path, clip=True)
        if mid.ticks_per_beat == 0: return None
        
        original_bpm = self._get_bpm(file_name, mid)
        scaling_factor = original_bpm / TARGET_BPM

        normalized_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
        for track in mid.tracks:
            new_track = mido.MidiTrack()
            for msg in track:
                msg_copy = msg.copy()
                if hasattr(msg_copy, 'time'):
                    msg_copy.time = int(round(msg_copy.time * scaling_factor))
                new_track.append(msg_copy)
            normalized_mid.tracks.append(new_track)
        
        ticks_per_beat = normalized_mid.ticks_per_beat
        ticks_per_bar = ticks_per_beat * 4
        ticks_per_step = ticks_per_bar / self.steps_per_bar
        
        if ticks_per_step == 0:
            print(f"Warning: ticks_per_step is zero for {file_name}. Skipping file.")
            return None

        events = []
        total_time_ticks = 0
        
        # --- ZMIANA: Wybór tylko trzeciej ścieżki (indeks 2) ---
        if len(normalized_mid.tracks) < 3:
            print(f"Warning: File {file_name} has fewer than 3 tracks ({len(normalized_mid.tracks)}). Skipping file.")
            return None

        piano_track = normalized_mid.tracks[2] # Wybieramy trzecią ścieżkę

        current_time_ticks = 0
        for msg in piano_track: # Iterujemy tylko po komunikatach z wybranej ścieżki
            current_time_ticks += msg.time
            total_time_ticks = max(total_time_ticks, current_time_ticks)
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append({'type': 'on', 'note': msg.note, 'time': current_time_ticks})
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                events.append({'type': 'off', 'note': msg.note, 'time': current_time_ticks})
        # --- KONIEC ZMIANY ---

        if not events: return None
        events.sort(key=lambda x: x['time'])
        
        total_steps = int(np.ceil(total_time_ticks / ticks_per_step))
        piano_roll = np.full((total_steps, 128), STATE_OFF, dtype=np.int8)
        
        active_notes = {}
        for event in events:
            step = int(round(event['time'] / ticks_per_step))
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
        
        return piano_roll, original_bpm

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns a tuple containing the piano roll sequence and its original BPM.
        If verbose is True, the file path is also included.
        """
        sequence = self.sequences[idx]
        bpm = torch.tensor(self.bpms[idx], dtype=torch.float32)
        
        if self.verbose:
            return sequence, bpm, self.file_paths[idx]
        else:
            # Poprawka: Zwracaj zawsze krotkę, aby zachować spójność
            return sequence, bpm
    
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

def prepare_dataloaders(split_ratios=(0.85, 0.15), seed: int = 42):
    torch.manual_seed(seed)
    print("Loading dataset with sliding window...")
    
    dataset = MidiDataset(
        midi_dir=DATASET_DIR,
        metadata_csv_path=METADATA_CSV_PATH,
        num_bars=NUM_BARS,
        steps_per_bar=STEPS_PER_BAR,
        use_sliding_window=USE_SLIDING_WINDOW,
        stride_in_bars=STRIDE_IN_BARS
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty after processing. Check MIDI files or processing logic.")

    train_size = int(split_ratios[0] * len(dataset))
    val_size = len(dataset) - train_size

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