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
    Autoregressive Hierarchical Decoder.
    Generates a piano roll step-by-step, feeding its own output back as input.
    Uses teacher forcing during training.
    """
    def __init__(self, latent_dim=LATENT_DIM, conductor_hidden_size=CONDUCTOR_HIDDEN_SIZE, 
                 decoder_hidden_size=DECODER_HIDDEN_SIZE, num_bars=NUM_BARS, 
                 steps_per_bar=STEPS_PER_BAR, num_layers=LSTM_LAYERS, 
                 note_embedding_dim=EMBEDDING_DIM):
        super().__init__()
        
        self.num_bars = num_bars
        self.steps_per_bar = steps_per_bar
        self.conductor_hidden_size = conductor_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.output_size_per_step = 128 * 3

        # --- Conductor Components ---
        self.z_to_conductor_initial = nn.Linear(latent_dim, conductor_hidden_size * num_layers * 2)
        self.conductor = nn.LSTM(
            input_size=1, 
            hidden_size=conductor_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # --- DecoderRNN Components ---
        self.note_projection = nn.Linear(INPUT_DIM, note_embedding_dim)
        decoder_input_size = conductor_hidden_size + note_embedding_dim
        
        self.conductor_to_decoder_initial = nn.Linear(conductor_hidden_size, decoder_hidden_size * num_layers * 2)
        self.decoder_rnn = nn.LSTM(
            input_size=decoder_input_size, 
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(decoder_hidden_size, self.output_size_per_step)

    def forward(self, z, target_pianoroll=None):
        batch_size = z.size(0)
        device = z.device

        # --- 1. Run the Conductor ---
        conductor_initial_flat = self.z_to_conductor_initial(z)
        conductor_initial_reshaped = conductor_initial_flat.view(batch_size, 2, self.num_layers, self.conductor_hidden_size)
        conductor_initial_permuted = conductor_initial_reshaped.permute(2, 0, 3, 1)
        h_c0 = conductor_initial_permuted[..., 0].contiguous()
        c_c0 = conductor_initial_permuted[..., 1].contiguous()
        
        conductor_input = torch.zeros(batch_size, self.num_bars, 1, device=device)
        conductor_embeddings, _ = self.conductor(conductor_input, (h_c0, c_c0))

        # --- 2. Run the DecoderRNN autoregressively ---
        all_step_logits = []
        prev_step_pianoroll = torch.zeros(batch_size, 128, device=device)

        for i in range(self.num_bars):
            current_bar_embedding = conductor_embeddings[:, i, :]
            
            decoder_initial_flat = self.conductor_to_decoder_initial(current_bar_embedding)
            decoder_initial_reshaped = decoder_initial_flat.view(batch_size, 2, self.num_layers, self.decoder_hidden_size)
            decoder_initial_permuted = decoder_initial_reshaped.permute(2, 0, 3, 1)
            h_d0 = decoder_initial_permuted[..., 0].contiguous()
            c_d0 = decoder_initial_permuted[..., 1].contiguous()
            decoder_hidden_state = (h_d0, c_d0)

            for t in range(self.steps_per_bar):
                prev_step_embedded = self.note_projection(prev_step_pianoroll)
                rnn_input = torch.cat([prev_step_embedded, current_bar_embedding], dim=-1).unsqueeze(1)
                output_hidden, decoder_hidden_state = self.decoder_rnn(rnn_input, decoder_hidden_state)
                step_logits = self.output_projection(output_hidden.squeeze(1))
                all_step_logits.append(step_logits)

                if target_pianoroll is not None:
                    # Teacher Forcing: Use the ground truth from the target tensor
                    current_step_index = i * self.steps_per_bar + t
                    prev_step_pianoroll = target_pianoroll[:, current_step_index, :].float()
                else:
                    # Generation: Use the model's own output
                    step_output_states = torch.argmax(step_logits.view(-1, 128, 3), dim=-1)
                    prev_step_pianoroll = step_output_states.float()
        
        final_logits = torch.stack(all_step_logits, dim=1)
        final_logits = final_logits.view(batch_size, self.num_bars * self.steps_per_bar, 128, 3)

        return final_logits

class LofiModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = HierarchicalDecoder()

        self.device = DEVICE
        self.to(self.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        # Pass the ground truth `x` to the decoder for teacher forcing during training
        recon_logits = self.decoder(z, target_pianoroll=x)
        return recon_logits, mu, logvar
    
    def load_weights(self, path: str):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            self.to(self.device) 
            print(f"Successfully loaded weights from {path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    
    def generate(self, output_path="generated.mid") -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(self.device)
            # Call the decoder without a target to trigger generation mode
            recon_logits = self.decoder(z)
            generated_pianoroll = torch.argmax(recon_logits, dim=-1).squeeze(0)
        if output_path:
            tensor_to_midi(generated_pianoroll, output_path, bpm=100)
        return generated_pianoroll

    def reconstruct(self, input_pianoroll: torch.Tensor, bpm, output_path="reconstructed/reconstructed.mid") -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            input_batch = input_pianoroll.unsqueeze(0).to(self.device)
            # The forward pass will handle teacher forcing correctly
            recon_logits, _, _ = self(input_batch)
            reconstructed_pianoroll = torch.argmax(recon_logits, dim=-1).squeeze(0)
        if output_path:
            tensor_to_midi(reconstructed_pianoroll, output_path, bpm=bpm)
        return reconstructed_pianoroll