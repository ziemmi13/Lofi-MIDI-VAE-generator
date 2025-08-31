# train.py (z dodaną logiką warm-up)

import torch
import os
from tqdm import tqdm

# Import our custom modules
from dataset import MidiDataset, prepare_dataloaders
from model import LofiModel
from loss import compute_loss
import torch.optim as optim
from config import *


def train(model, verbose=True):
    model.to(DEVICE)
    print(f"Using device: {DEVICE}\n")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    train_dataloader, val_dataloader = prepare_dataloaders()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("-----------------------------")
    print("----- Starting training -----")
    print("-----------------------------\n")
    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Training]")
        for batch in pbar:
            batch = batch.to(DEVICE)

            # --- UPDATED KL ANNEALING LOGIC ---
            if epoch <= BETA_WARMUP_EPOCHS:
                # During warmup, beta is 0, focusing only on reconstruction.
                beta = 0.0
            else:
                # After warmup, start annealing beta as before.
                # We subtract the warmup steps from global_step for a smoother start.
                current_anneal_step = global_step - (len(train_dataloader) * BETA_WARMUP_EPOCHS)
                beta = min(BETA_END,
                           BETA_START + (BETA_END - BETA_START) * current_anneal_step / BETA_ANNEAL_STEPS)
            # --- END OF UPDATE ---
            
            recon_logits, mu, logvar = model(batch)
            
            losses = compute_loss(recon_logits, batch, mu, logvar, beta)
            total_loss = losses['total_loss']
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
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
                losses = compute_loss(recon_logits, batch, mu, logvar, BETA_END)
                val_loss_total += losses['total_loss'].item()

        # --- Epoch Summary and Logging ---
        avg_train_loss = total_loss.item() 
        avg_val_loss = val_loss_total / len(val_dataloader)
        print(f"\nEpoch {epoch} Summary: Last Train Batch Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        
        # --- Save Checkpoint ---
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"lofi_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        if verbose:
            random_num = torch.randint(0, len(val_dataloader.dataset), (1,)).item()
            random_tensor = val_dataloader.dataset[random_num]
            print(f"\nReconstructing random validation sample #{random_num}:")
            model.reconstruct(random_tensor)
            print('_' * 60, "\n")

    print("Training finished.")