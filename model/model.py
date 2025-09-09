import torch
import torch.nn as nn
from config import *
from utils import *

class Encoder(nn.Module):
    """
    Encodes a piano roll sequence into the parameters of a latent distribution
    (mu and logvar).
    """
    def __init__(self, input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM, hidden_size=ENCODER_HIDDEN_SIZE, latent_dim=LATENT_DIM, num_layers=LSTM_LAYERS):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc_mu = nn.Linear(hidden_size * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_dim)

    def forward(self, x):
        x = x.float()
        x_projected = self.input_projection(x)
        _, (h_n, _) = self.lstm(x_projected)
        h_n_last_layer = torch.cat([h_n[-2,:,:], h_n[-1,:,:]], dim=-1)
        mu = self.fc_mu(h_n_last_layer)
        logvar = self.fc_logvar(h_n_last_layer)
        return mu, logvar

class HierarchicalDecoder(nn.Module):
    """
    Decodes a latent vector `z` into a piano roll sequence using a two-level
    hierarchical structure (Conductor and DecoderRNN).
    """
    ### CHANGED ### - Added 'num_pitches=NUM_PITCHES' to the function definition
    def __init__(self, latent_dim=LATENT_DIM, conductor_hidden_size=CONDUCTOR_HIDDEN_SIZE, 
                 decoder_hidden_size=DECODER_HIDDEN_SIZE, num_bars=NUM_BARS, 
                 steps_per_bar=STEPS_PER_BAR, num_layers=LSTM_LAYERS, num_pitches=NUM_PITCHES):
        super().__init__()
        
        self.num_bars = num_bars
        self.steps_per_bar = steps_per_bar
        self.conductor_hidden_size = conductor_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        
        ### CHANGED ### - Use the 'num_pitches' parameter, not a hardcoded value
        self.num_pitches = num_pitches
        self.output_size_per_step = self.num_pitches * 3

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
        
        ### CHANGED ### - Use 'self.num_pitches' in the final reshape
        final_logits = logits.view(batch_size, self.num_bars * self.steps_per_bar, self.num_pitches, 3)

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
        
        ### CHANGED ### - Pass the NUM_PITCHES parameter to the decoder
        self.decoder = HierarchicalDecoder(
            latent_dim=LATENT_DIM,
            conductor_hidden_size=CONDUCTOR_HIDDEN_SIZE,
            decoder_hidden_size=DECODER_HIDDEN_SIZE,
            num_bars=NUM_BARS,
            steps_per_bar=STEPS_PER_BAR,
            num_layers=LSTM_LAYERS,
            num_pitches=NUM_PITCHES
        )

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
            tensor_to_midi(generated_pianoroll, output_path, bpm=80)
        return generated_pianoroll

    def reconstruct(self, input_pianoroll: torch.Tensor, output_path="reconstructed/reconstructed.mid") -> torch.Tensor:
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            input_batch  = input_pianoroll.unsqueeze(0).to(device)
            recon_logits, _, _ = self(input_batch)
            reconstructed_pianoroll = torch.argmax(recon_logits, dim=-1).squeeze(0)
        if output_path:
            tensor_to_midi(reconstructed_pianoroll, output_path, bpm=80)
        return reconstructed_pianoroll

