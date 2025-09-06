import torch
import os
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from IPython.display import display

# Import our custom modules
from dataset_finetuning import MidiDataset, prepare_dataloaders
from model import LofiModel
from loss import compute_loss
from utils import calculate_class_weights
from finetune_config import * 
from train_utils import EarlyStopping, setup_commet_loger

def finetune(model, freeze_encoder=True, early_stopping=False, experiment_name=None, verbose=True):
    """
    General purpose training function for the LofiModel.
    Can be used for training from scratch or for fine-tuning.
    """
    model.to(DEVICE)
    print(f"Using device: {DEVICE}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    train_dataloader, val_dataloader = prepare_dataloaders()
    
    class_weights = calculate_class_weights(train_dataloader).to(DEVICE)
    
    # --- Flexible Optimizer Setup ---
    if freeze_encoder:
        print("Encoder layers are FROZEN. Training only the decoder.")
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.decoder.parameters(), lr=LEARNING_RATE)
    else:
        print("Training all model layers.")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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
        early_stopper = EarlyStopping(patience=15, path=os.path.join(CHECKPOINT_DIR, "best_finetuned_model.pt"), verbose=True)

    print("-----------------------------")
    print("----- Starting training -----")
    print("-----------------------------\n")

    global_step = 0
    # Use NUM_EPOCHS from the config file
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        if freeze_encoder: # Make sure encoder stays in eval mode if frozen
            model.encoder.eval()

        train_loss_total = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Fine-Tuning]")
        for batch in pbar:
            # Assumes the dataset returns only the piano roll tensor
            batch = batch.to(DEVICE)
            
            # For fine-tuning, we use a constant beta
            beta = BETA_END
            
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
            
            if experiment:
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
    print("Fine-tuning finished.")

def run_finetune(experiment_name):
    print("Creating model instance...")
    model = LofiModel()

    print(f"Loading pre-trained weights from {PRETRAINED_CHECKPOINT_PATH}...")
    if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        model.load_weights(PRETRAINED_CHECKPOINT_PATH)
    else:
        print(f"FATAL: Pre-trained checkpoint not found at {PRETRAINED_CHECKPOINT_PATH}. Cannot start fine-tuning.")

    if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        try:
            finetune(model, experiment_name="Lofi Model - Finetuning")
        except KeyboardInterrupt:
            print("\nFinetuning interrupted by user.")