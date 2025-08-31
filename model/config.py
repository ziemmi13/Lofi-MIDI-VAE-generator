import torch

"DATASET"
DATASET_DIR = r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\PZ#2\dataset\maestro-v3.0.0"
NUM_BARS= 4
STEPS_PER_BAR = 32

"MODEL"
INPUT_DIM = 128
EMBEDDING_DIM = 64
ENCODER_HIDDEN_SIZE = 128
LATENT_DIM = 128
CONDUCTOR_HIDDEN_SIZE = 128
DECODER_HIDDEN_SIZE = 128
LSTM_LAYERS = 2


"""TRAIN"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Training Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

# --- VAE Loss Control (KL Annealing) ---
BETA_START = 0.0
BETA_END = 0.2
BETA_ANNEAL_STEPS = 20000

# --- Dataloader ---
NUM_WORKERS = 24

# --- Checkpoints and Logging ---
CHECKPOINT_DIR = "checkpoints/"
LOG_DIR = "runs/"
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 1
