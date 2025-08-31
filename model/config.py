# config.py

import torch

class DataConfig:
    """Configuration for data loading and preprocessing."""
    midi_dir = r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\PZ#2\dataset\maestro-v3.0.0"
    num_bars = 4
    steps_per_bar = 32

class ModelConfig:
    """Configuration for the VAE model architecture."""
    input_dim = 128
    embedding_dim = 128
    encoder_hidden_size = 512
    latent_dim = 256
    conductor_hidden_size = 512
    decoder_hidden_size = 512
    num_layers = 2
    
    num_bars = DataConfig.num_bars
    steps_per_bar = DataConfig.steps_per_bar
    
class TrainConfig:
    """Configuration for the training process."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Training Hyperparameters ---
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 100
    
    # --- VAE Loss Control (KL Annealing) ---
    beta_start = 0.0
    beta_end = 0.2
    beta_anneal_steps = 20000
    
    # --- Dataloader ---
    num_workers = 4
    
    # --- Checkpoints and Logging ---
    checkpoint_dir = "checkpoints/"
    log_dir = "runs/"
    log_interval = 100
    checkpoint_interval = 5

class Config:
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.train = TrainConfig()

config = Config()