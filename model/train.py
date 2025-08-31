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

    # TODO: Add TensorBoard SummaryWriter here for logging if desired
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter(config.train.log_dir)
    
    train_dataloader, val_dataloader = prepare_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("-----------------------------")
    print("----- Starting training -----")
    print("-----------------------------\n")
    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss_total, train_loss_recon, train_loss_kl = 0, 0, 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Training]")
        for batch in pbar:
            # Move data to the configured device
            batch = batch.to(DEVICE)

            # --- KL Annealing ---
            # Calculate current beta for KL annealing
            beta = min(BETA_END,
                       BETA_START + (BETA_END - BETA_START) * global_step / BETA_ANNEAL_STEPS)
            
            # Forward pass
            recon_logits, mu, logvar = model(batch)
            
            # Calculate loss
            losses = compute_loss(recon_logits, batch, mu, logvar, beta)
            total_loss = losses['total_loss']
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses for logging
            train_loss_total += total_loss.item()
            train_loss_recon += losses['recon_loss'].item()
            train_loss_kl += losses['kl_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}", 
                'Recon': f"{losses['recon_loss'].item():.4f}", 
                'KL': f"{losses['kl_loss'].item():.4f}",
                'beta': f"{beta:.4f}"
            })

            global_step += 1

        # --- Validation Phase ---
        model.eval()
        val_loss_total, val_loss_recon, val_loss_kl = 0, 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(DEVICE)
                recon_logits, mu, logvar = model(batch)
                
                # Use the final beta value for validation
                losses = compute_loss(recon_logits, batch, mu, logvar, BETA_END)
                
                val_loss_total += losses['total_loss'].item()
                val_loss_recon += losses['recon_loss'].item()
                val_loss_kl += losses['kl_loss'].item()

        # --- Epoch Summary and Logging ---
        avg_train_loss = train_loss_total / len(train_dataloader)
        avg_val_loss = val_loss_total / len(val_dataloader)
        print(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # TODO: Log metrics to TensorBoard here
        # writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
        # writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
        
        # --- Save Checkpoint ---
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"lofi_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        if verbose:
            random_num = torch.randint(0, len(train_dataloader.dataset), (1,)).item()
            random_tensor = train_dataloader.dataset[random_num]
            print(f"\nEpoch {epoch+1}. Reconstructing random sample {random_num}:")
            model.reconstruct(random_tensor)
            print('_' * 60, "\n")

    # TODO: Close the TensorBoard writer
    # writer.close()
    print("Training finished.")
