import torch
import torch.nn.functional as F
from config import KL_FREE_BITS

def compute_loss(recon_logits, target_pianoroll, mu, logvar, beta, class_weights=None):
    """
    Calculates the VAE loss for the 3-state piano roll.
    """
    # Reshape logits: (batch * steps * 128, 3)
    recon_logits_flat = recon_logits.view(-1, 3) 
    # Reshape targets: (batch * steps * 128)
    target_pianoroll_flat = target_pianoroll.view(-1)
    
    # Use CrossEntropyLoss for multi-class (3 states) classification
    recon_loss = F.cross_entropy(recon_logits_flat, target_pianoroll_flat, 
                                 weight=class_weights, reduction='mean')
    
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss_kl_free_bits = torch.relu(kl_div - KL_FREE_BITS)

    total_loss = recon_loss + beta * loss_kl_free_bits 
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_div
    }