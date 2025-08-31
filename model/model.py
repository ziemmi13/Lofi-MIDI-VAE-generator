import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encodes a piano roll sequence into the parameters of a latent distribution
    (mu and logvar).
    """
    def __init__(self, input_dim=128, embedding_dim=64, hidden_size=256, latent_dim=128, num_layers=2):
        super().__init__()
        
        # Input projection "embedding" layer
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        
        # Bidirectional LSTM to process the sequence
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layers to predict mu and logvar
        self.fc_mu = nn.Linear(hidden_size * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_dim)

    def forward(self, x):
        # x shape: (batch_size, num_steps, 128)
        
        # Ensure the same type of input and weights
        x = x.float()

        # Project the 128-dim piano roll slice to a denser embedding
        x_projected = self.input_projection(x) # -> (batch_size, num_steps, embedding_dim)
        
        # Pass through LSTM. We only need the final hidden state.
        _, (h_n, _) = self.lstm(x_projected) # h_n shape: (num_layers*2, batch_size, hidden_size)
        
        # Concatenate the final hidden states from both directions of the last layer
        h_n_last_layer = torch.cat([h_n[-2,:,:], h_n[-1,:,:]], dim=-1) # -> (batch_size, hidden_size*2)
        
        # Calculate mu and logvar
        mu = self.fc_mu(h_n_last_layer)
        logvar = self.fc_logvar(h_n_last_layer)
        
        return mu, logvar

# Wklej tę finalną, poprawioną klasę do pliku model.py

class HierarchicalDecoder(nn.Module):
    """
    Decodes a latent vector `z` into a piano roll sequence using a two-level
    hierarchical structure (Conductor and DecoderRNN).
    """
    def __init__(self, latent_dim=128, conductor_hidden_size=256, decoder_hidden_size=256, 
                 num_bars=2, steps_per_bar=32, num_layers=2):
        super().__init__()
        
        self.num_bars = num_bars
        self.steps_per_bar = steps_per_bar
        self.conductor_hidden_size = conductor_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.output_size_per_step = 128 * 3

        self.z_to_conductor_initial = nn.Linear(latent_dim, conductor_hidden_size * num_layers * 2)
        self.conductor = nn.LSTM(
            input_size=1, 
            hidden_size=conductor_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.conductor_to_decoder_initial = nn.Linear(conductor_hidden_size, decoder_hidden_size * num_layers * 2)
        self.decoder_rnn = nn.LSTM(
            input_size=1, 
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(decoder_hidden_size, self.output_size_per_step)

    def forward(self, z):
        batch_size = z.size(0)

        # --- 1. Run the Conductor ---
        conductor_initial_flat = self.z_to_conductor_initial(z)
        conductor_initial_reshaped = conductor_initial_flat.view(batch_size, 2, self.num_layers, self.conductor_hidden_size)
        conductor_initial_permuted = conductor_initial_reshaped.permute(2, 0, 3, 1)

        # --- FINAL FIX: Add .contiguous() after slicing ---
        h_c0 = conductor_initial_permuted[..., 0].contiguous()
        c_c0 = conductor_initial_permuted[..., 1].contiguous()
        # --- END OF FIX ---

        conductor_input = torch.zeros(batch_size, self.num_bars, 1, device=z.device)
        conductor_embeddings, _ = self.conductor(conductor_input, (h_c0, c_c0))

        # --- 2. Run the DecoderRNN ---
        all_bar_outputs = []
        for i in range(self.num_bars):
            current_bar_embedding = conductor_embeddings[:, i, :]
            
            decoder_initial_flat = self.conductor_to_decoder_initial(current_bar_embedding)
            decoder_initial_reshaped = decoder_initial_flat.view(batch_size, 2, self.num_layers, self.decoder_hidden_size)
            decoder_initial_permuted = decoder_initial_reshaped.permute(2, 0, 3, 1)

            # --- FINAL FIX: Add .contiguous() after slicing ---
            h_d0 = decoder_initial_permuted[..., 0].contiguous()
            c_d0 = decoder_initial_permuted[..., 1].contiguous()
            # --- END OF FIX ---

            decoder_input = torch.zeros(batch_size, self.steps_per_bar, 1, device=z.device)
            bar_output_hidden, _ = self.decoder_rnn(decoder_input, (h_d0, c_d0))
            all_bar_outputs.append(bar_output_hidden)
        
        concatenated_outputs = torch.cat(all_bar_outputs, dim=1)

        # --- 3. Project to final output shape ---
        logits = self.output_projection(concatenated_outputs)
        final_logits = logits.view(batch_size, self.num_bars * self.steps_per_bar, 128, 3)

        return final_logits

class LofiModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        self.encoder = Encoder(
            input_dim=config['input_dim'],
            embedding_dim=config['embedding_dim'],
            hidden_size=config['encoder_hidden_size'],
            latent_dim=config['latent_dim'],
            num_layers=config['num_layers']
        )
        
        self.decoder = HierarchicalDecoder(
            latent_dim=config['latent_dim'],
            conductor_hidden_size=config['conductor_hidden_size'],
            decoder_hidden_size=config['decoder_hidden_size'],
            num_bars=config['num_bars'],
            steps_per_bar=config['steps_per_bar'],
            num_layers=config['num_layers']
        )

    def reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Defines the forward pass of the VAE.

        Args:
            x (torch.Tensor): The input piano roll tensor. 
                              Shape: (batch_size, num_steps, 128)

        Returns:
            recon_logits (torch.Tensor): The reconstructed piano roll logits.
                                         Shape: (batch_size, num_steps, 128, 3)
            mu (torch.Tensor): The mean of the latent distribution.
                               Shape: (batch_size, latent_dim)
            logvar (torch.Tensor): The log variance of the latent distribution.
                                   Shape: (batch_size, latent_dim)
        """
        # 1. Encode the input to get latent distribution parameters
        mu, logvar = self.encoder(x)
        
        # 2. Sample from the latent distribution using the reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode the latent vector to reconstruct the input
        recon_logits = self.decoder(z)
        
        return recon_logits, mu, logvar

