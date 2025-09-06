# finetune_config.py

import torch

"DATASET"
DATASET_DIR = r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\PZ#2\dataset\lofi-dataset" 
METADATA_CSV_PATH = r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\PZ#2\dataset\midi_metadata_dla_3_instrumentow.csv"
NUM_BARS= 4
STEPS_PER_BAR = 16
USE_SLIDING_WINDOW = True
STRIDE_IN_BARS = 1
NUM_WORKERS = 0
TARGET_BPM = 80

"MODEL"
INPUT_DIM = 128
EMBEDDING_DIM = 128
ENCODER_HIDDEN_SIZE = 512
LATENT_DIM = 512
CONDUCTOR_HIDDEN_SIZE = 512
DECODER_HIDDEN_SIZE = 512
LSTM_LAYERS = 2

PRETRAINED_CHECKPOINT_PATH = r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\PZ#2\model\checkpoints\finetuned_lofi_model\lofi_model_epoch_165.pth"

"VAE PARAMS"
BETA_START = 0.05
BETA_END = 0.1   # Keep beta constant
BETA_ANNEAL_STEPS = 10000 # Effectively disable annealing
BETA_WARMUP_EPOCHS = 100  # Disable warmup
KL_FREE_BITS = 0.5

"""TRAIN"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

LEARNING_RATE = 1e-5 

NUM_EPOCHS = 300

"CHECKPOINTS AND LOGGING"
CHECKPOINT_DIR = "checkpoints/finetuned_lofi_model/"
LOG_DIR = "runs/finetuned_lofi_model/"
CHECKPOINT_INTERVAL = 5
LOG_INTERVAL = 10