# train.py

import torch
import torch.optim as optim
import os
from tqdm import tqdm

# Import our custom modules
from dataset import MidiDataset
from model import LofiModel
from loss import compute_loss
from config import config

def train():

    print(f"Using device: {config.train.device}\n")

    # Create directories for checkpoints and logs if they don't exist
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    os.makedirs(config.train.log_dir, exist_ok=True)

    # TODO: Add TensorBoard SummaryWriter here for logging if desired
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter(config.train.log_dir)
    
    print("Loading dataset...")
    full_dataset = MidiDataset(
        midi_dir=config.data.midi_dir,
        num_bars=config.data.num_bars,
        steps_per_bar=config.data.steps_per_bar
    )
    dataloaders = full_dataset.prepare_dataloaders(batch_size=config.train.batch_size, num_workers=config.train.num_workers)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    print("Finished loading dataset.\n")
    
    print("Initializing model...")
    model_params = {
        key: getattr(config.model, key)
        for key in dir(config.model)
        if not key.startswith('__') and not isinstance(getattr(config.model, key), type)
    }
    
    model = LofiModel(model_params).to(config.train.device)
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
    
    print(model)
    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters.\n")

    print("-----------------------------")
    print("----- Starting training -----")
    print("-----------------------------\n")
    global_step = 0
    for epoch in range(1, config.train.num_epochs + 1):
        # --- Training Phase ---
        model.train()
        train_loss_total, train_loss_recon, train_loss_kl = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.train.num_epochs} [Training]")
        for batch in pbar:
            # Move data to the configured device
            batch = batch.to(config.train.device)

            # --- KL Annealing ---
            # Calculate current beta for KL annealing
            beta = min(config.train.beta_end, 
                       config.train.beta_start + (config.train.beta_end - config.train.beta_start) * global_step / config.train.beta_anneal_steps)
            
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
            for batch in val_loader:
                batch = batch.to(config.train.device)
                recon_logits, mu, logvar = model(batch)
                
                # Use the final beta value for validation
                losses = compute_loss(recon_logits, batch, mu, logvar, config.train.beta_end)
                
                val_loss_total += losses['total_loss'].item()
                val_loss_recon += losses['recon_loss'].item()
                val_loss_kl += losses['kl_loss'].item()

        # --- Epoch Summary and Logging ---
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # TODO: Log metrics to TensorBoard here
        # writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
        # writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
        
        # --- Save Checkpoint ---
        if epoch % config.train.checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.train.checkpoint_dir, f"lofi_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # TODO: Close the TensorBoard writer
    # writer.close()
    print("Training finished.")

if __name__ == '__main__':
    train()