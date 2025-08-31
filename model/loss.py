import torch
import torch.nn.functional as F

def compute_loss(recon_logits, target_pianoroll, mu, logvar, beta, class_weights=None):
    """
    Calculates the VAE loss, now with optional class weighting for reconstruction.
    """
    recon_logits_flat = recon_logits.view(-1, 3) 
    target_pianoroll_flat = target_pianoroll.view(-1)
    
    # Use the weights in the cross entropy loss function
    recon_loss = F.cross_entropy(recon_logits_flat, target_pianoroll_flat, 
                                 weight=class_weights, reduction='mean')
    
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_div
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_div
    }