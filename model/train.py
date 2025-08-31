# train.py (wersja finalna)

import torch
import os
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import our custom modules
from dataset import prepare_dataloaders
from model import LofiModel
from loss import compute_loss
from utils import calculate_class_weights
from config import *


def train(model, verbose=True):
    model.to(DEVICE)
    print(f"Using device: {DEVICE}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    train_dataloader, val_dataloader = prepare_dataloaders()
    
    class_weights = calculate_class_weights(train_dataloader).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("-----------------------------")
    print("----- Starting training -----")
    print("-----------------------------\n")

    
    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        # Reset accumulated train losses for each epoch
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
            
            # Accumulate total train loss for the epoch average
            train_loss_total += total_loss.item()
            
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}", 
                'Recon': f"{losses['recon_loss'].item():.4f}", 
                'KL': f"{losses['kl_loss'].item():.4f}",
                'beta': f"{beta:.4f}"
            })

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
        # --- CORRECTED LINE ---
        # Calculate the average train loss for the entire epoch
        avg_train_loss = train_loss_total / len(train_dataloader)
        avg_val_loss = val_loss_total / len(val_dataloader)
        print(f"\nEpoch {epoch} Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # --- Save Checkpoint ---
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"lofi_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        if verbose and epoch % 10:
            random_num = torch.randint(0, len(val_dataloader.dataset), (1,)).item()
            random_tensor = val_dataloader.dataset[random_num]
            print(f"\nReconstructing random validation sample #{random_num}:")
            model.reconstruct(random_tensor)
            print('_' * 60, "\n")

    print("Training finished.")