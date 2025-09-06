import torch
import torch.nn as nn
from dataset import MidiDataset
from config import *
from utils import *

class Encoder(nn.Module):
    """
    Encodes a piano roll sequence into the parameters of a latent distribution
    (mu and logvar).
    """
    def __init__(self, input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM, hidden_size=ENCODER_HIDDEN_SIZE, latent_dim=LATENT_DIM, num_layers=LSTM_LAYERS):
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

class HierarchicalDecoder(nn.Module):
    """
    Decodes a latent vector `z` into a piano roll sequence using a two-level
    hierarchical structure (Conductor and DecoderRNN).
    """
    def __init__(self, latent_dim=LATENT_DIM, conductor_hidden_size=CONDUCTOR_HIDDEN_SIZE, decoder_hidden_size=DECODER_HIDDEN_SIZE, 
                 num_bars=NUM_BARS, steps_per_bar=STEPS_PER_BAR, num_layers=LSTM_LAYERS):
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
        self.device = DEVICE

    def forward(self, z):
        batch_size = z.size(0)

        # --- 1. Run the Conductor ---
        conductor_initial_flat = self.z_to_conductor_initial(z)
        conductor_initial_reshaped = conductor_initial_flat.view(batch_size, 2, self.num_layers, self.conductor_hidden_size)
        conductor_initial_permuted = conductor_initial_reshaped.permute(2, 0, 3, 1)

        h_c0 = conductor_initial_permuted[..., 0].contiguous()
        c_c0 = conductor_initial_permuted[..., 1].contiguous()

        conductor_input = torch.zeros(batch_size, self.num_bars, 1, device=self.device)
        conductor_embeddings, _ = self.conductor(conductor_input, (h_c0, c_c0))

        # --- 2. Run the DecoderRNN ---
        all_bar_outputs = []
        for i in range(self.num_bars):
            current_bar_embedding = conductor_embeddings[:, i, :]
            
            decoder_initial_flat = self.conductor_to_decoder_initial(current_bar_embedding)
            decoder_initial_reshaped = decoder_initial_flat.view(batch_size, 2, self.num_layers, self.decoder_hidden_size)
            decoder_initial_permuted = decoder_initial_reshaped.permute(2, 0, 3, 1)

            h_d0 = decoder_initial_permuted[..., 0].contiguous()
            c_d0 = decoder_initial_permuted[..., 1].contiguous()

            decoder_input = torch.zeros(batch_size, self.steps_per_bar, 1, device=self.device)
            bar_output_hidden, _ = self.decoder_rnn(decoder_input, (h_d0, c_d0))
            all_bar_outputs.append(bar_output_hidden)
        
        concatenated_outputs = torch.cat(all_bar_outputs, dim=1)

        # --- 3. Project to final output shape ---
        logits = self.output_projection(concatenated_outputs)
        final_logits = logits.view(batch_size, self.num_bars * self.steps_per_bar, 128, 3)

        return final_logits

class LofiModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder(
            input_dim=INPUT_DIM,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=ENCODER_HIDDEN_SIZE,
            latent_dim=LATENT_DIM,
            num_layers=LSTM_LAYERS
        )
        
        self.decoder = HierarchicalDecoder(
            latent_dim=LATENT_DIM,
            conductor_hidden_size=CONDUCTOR_HIDDEN_SIZE,
            decoder_hidden_size=DECODER_HIDDEN_SIZE,
            num_bars=NUM_BARS,
            steps_per_bar=STEPS_PER_BAR,
            num_layers=LSTM_LAYERS
        )

        # The model's device is set once upon creation.
        self.device = DEVICE
        self.to(self.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decoder(z)
        return recon_logits, mu, logvar
    
    def load_weights(self, path: str):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            self.to(self.device) 
            print(f"Successfully loaded weights from {path}")
        except FileNotFoundError:
            print(f"Error: Weights not found at {path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    
    def generate(self, output_path="generated.mid") -> torch.Tensor:
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(device)
            recon_logits = self.decoder(z)
            generated_pianoroll = torch.argmax(recon_logits, dim=-1).squeeze(0)

        if output_path:
            tensor_to_midi(generated_pianoroll, output_path)
        return generated_pianoroll

    def reconstruct(self, input_pianoroll: torch.Tensor, output_path="reconstructed/reconstructed.mid") -> torch.Tensor:
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            input_batch = input_pianoroll.unsqueeze(0).to(device)
            recon_logits, _, _ = self(input_batch)
            reconstructed_pianoroll = torch.argmax(recon_logits, dim=-1).squeeze(0)
        if output_path:
            tensor_to_midi(reconstructed_pianoroll, output_path)
        return reconstructed_pianoroll

