import torch
import torch.nn.functional as F

def vae_loss_function(recon_logits, target_pianoroll, mu, logvar, beta):
    # Reshape logits: (batch * steps * 128, 3)
    recon_logits_flat = recon_logits.view(-1, 3) 
    # Reshape targets: (batch * steps * 128)
    target_pianoroll_flat = target_pianoroll.view(-1)
    
    recon_loss = F.cross_entropy(recon_logits_flat, target_pianoroll_flat, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_div
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_div
    }