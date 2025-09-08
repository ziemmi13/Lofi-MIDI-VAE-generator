# train.py

import torch
import os
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from IPython.display import display

# Import our custom modules
from dataset import MidiDataset, prepare_dataloaders
from model import LofiModel
from loss import compute_loss
from utils import calculate_class_weights, visualize_latent_space
# from config import * 
from model.config_finetuning import *
from train_utils import EarlyStopping, setup_commet_loger


def train(model, early_stopping=False, experiment_name=None, verbose=True):
    """
    Main function to train the LofiModel.
    
    Args:
        model (LofiModel): The model instance to be trained.
        early_stopping (bool): If True, enables early stopping.
        experiment_name (str, optional): The name for the Comet.ml experiment. If None, logging is disabled.
        verbose (bool): If True, shows reconstructions during training.
    """
    model.to(DEVICE)
    print(f"Using device: {DEVICE}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    train_dataloader, val_dataloader = prepare_dataloaders()
    
    # Calculate class weights for the loss function to handle data imbalance
    class_weights = calculate_class_weights(train_dataloader).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- COMET & EARLY STOPPING SETUP ---
    experiment = None
    if experiment_name:
        experiment = setup_commet_loger(experiment_name)
        # Log all hyperparameters from the config file for reproducibility
        hyperparams = {
            "dataset_dir": DATASET_DIR,
            "num_bars": NUM_BARS, "steps_per_bar": STEPS_PER_BAR,
            "use_sliding_window": USE_SLIDING_WINDOW, "stride_in_bars": STRIDE_IN_BARS,
            "model_input_dim": INPUT_DIM, "model_embedding_dim": EMBEDDING_DIM,
            "model_encoder_hidden_size": ENCODER_HIDDEN_SIZE, "model_latent_dim": LATENT_DIM,
            "model_conductor_hidden_size": CONDUCTOR_HIDDEN_SIZE, "model_decoder_hidden_size": DECODER_HIDDEN_SIZE,
            "model_lstm_layers": LSTM_LAYERS,
            "vae_beta_start": BETA_START, "vae_beta_end": BETA_END,
            "vae_beta_anneal_steps": BETA_ANNEAL_STEPS, "vae_beta_warmup_epochs": BETA_WARMUP_EPOCHS,
            "vae_kl_free_bits": KL_FREE_BITS,
            "train_device": DEVICE, "train_batch_size": BATCH_SIZE,
            "train_learning_rate": LEARNING_RATE, "train_num_epochs": NUM_EPOCHS
        }
        experiment.log_parameters(hyperparams)

    early_stopper = None
    if early_stopping:
        early_stopper = EarlyStopping(patience=30, path=os.path.join(CHECKPOINT_DIR, "best_model.pt"), verbose=True)
    
    print("-----------------------------")
    print("----- Starting training -----")
    print("-----------------------------\n")

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_total = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Training]")
        for batch in pbar:
            batch = batch.to(DEVICE)

            if epoch <= BETA_WARMUP_EPOCHS:
                beta = 0.0
            else:
                current_anneal_step = global_step - (len(train_dataloader) * BETA_WARMUP_EPOCHS)
                beta = min(BETA_END,
                           BETA_START + (BETA_END - BETA_START) * current_anneal_step / BETA_ANNEAL_STEPS)
            
            recon_logits, mu, logvar = model(batch)
            losses = compute_loss(recon_logits, batch, mu, logvar, beta, class_weights)
            total_loss = losses['total_loss']
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_total += total_loss.item()
            
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}", 'Recon': f"{losses['recon_loss'].item():.4f}", 
                'KL': f"{losses['kl_loss'].item():.4f}", 'beta': f"{beta:.4f}"
            })
            
            if experiment and global_step % LOG_INTERVAL == 0:
                experiment.log_metric("train_batch_loss", total_loss.item(), step=global_step)
                experiment.log_metric("train_batch_recon_loss", losses['recon_loss'].item(), step=global_step)
                experiment.log_metric("train_batch_kl_loss", losses['kl_loss'].item(), step=global_step)
                experiment.log_metric("beta", beta, step=global_step)

            global_step += 1

        # --- Validation Phase ---
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(DEVICE)
                recon_logits, mu, logvar = model(batch)
                losses = compute_loss(recon_logits, batch, mu, logvar, BETA_END, class_weights)
                val_loss_total += losses['total_loss'].item()

        # --- Epoch Summary and Logging ---
        avg_train_loss = train_loss_total / len(train_dataloader)
        avg_val_loss = val_loss_total / len(val_dataloader)
        print(f"\nEpoch {epoch} Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        
        if experiment:
            experiment.log_metric("avg_train_loss", avg_train_loss, epoch=epoch)
            experiment.log_metric("avg_val_loss", avg_val_loss, epoch=epoch)
            experiment.log_metric("learning_rate", optimizer.param_groups[0]['lr'], epoch=epoch)

        scheduler.step(avg_val_loss)
        
        if early_stopping:
            early_stopper(avg_val_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break
        
        if epoch % CHECKPOINT_INTERVAL == 0 and not early_stopping:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"lofi_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == NUM_EPOCHS):
            print(f"\nGenerating visualization for epoch {epoch}:")
            random_num = torch.randint(0, len(val_dataloader.dataset), (1,)).item()
            original_tensor = val_dataloader.dataset[random_num]
            reconstructed_tensor = model.reconstruct(original_tensor)
            
            fig, axes = plt.subplots(2, 1, figsize=(16, 12))
            fig.suptitle(f"Reconstruction at Epoch {epoch}", fontsize=20)
            
            MidiDataset.visualize(original_tensor, title="Original", ax=axes[0], show_plot=False)
            MidiDataset.visualize(reconstructed_tensor, title="Reconstructed", ax=axes[1], show_plot=False)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if experiment:
                experiment.log_figure(figure=fig, figure_name=f"Reconstruction_Epoch_{epoch}", overwrite=True)
            
            display(fig)
            plt.close(fig)
            print('_' * 60, "\n")
    
    if experiment:
        experiment.end()
    print("Training finished.")