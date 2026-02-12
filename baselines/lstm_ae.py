"""LSTM-AE: LSTM Autoencoder for Anomaly Detection.

LSTM autoencoder with a deterministic latent bottleneck (no variational component).

Architecture:
- Encoder: Bidirectional LSTM -> concat fwd+bwd (2*hidden_dim) -> Linear -> latent_dim
- Decoder: Unidirectional LSTM, init hidden/cell from linear projections of latent
- Loss: MSE reconstruction only (NO KL divergence)

Key difference from LSTM-VAE: single deterministic latent, no reparameterization.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaselineModel


class LSTMAE(BaselineModel):
    """LSTM Autoencoder with deterministic latent bottleneck."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_measurement_vars: int = None,
    ):
        super().__init__(
            name="LSTM-AE",
            n_features=n_features,
            window_size=window_size,
            n_measurement_vars=n_measurement_vars,
        )

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder: Bidirectional LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project concatenated fwd+bwd to deterministic latent
        self.fc_latent = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder init from latent
        self.fc_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_c = nn.Linear(latent_dim, hidden_dim * num_layers)

        # Decoder: Unidirectional LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Reconstruction [batch, n_features, window_size]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        x_seq = x.transpose(1, 2)  # [batch, seq_len, n_features]

        # Encode
        _, (h_n, _) = self.encoder_lstm(x_seq)

        # h_n: [num_layers*2, batch, hidden_dim]
        # Take last layer fwd+bwd
        h_forward = h_n[-2]  # [batch, hidden_dim]
        h_backward = h_n[-1]  # [batch, hidden_dim]
        h_combined = torch.cat([h_forward, h_backward], dim=-1)  # [batch, hidden_dim*2]

        # Deterministic latent (no reparameterization)
        z = self.fc_latent(h_combined)  # [batch, latent_dim]

        # Decode: project latent to decoder hidden/cell states
        h0 = self.fc_h(z).view(batch_size, self.num_layers, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()  # [num_layers, batch, hidden_dim]
        c0 = self.fc_c(z).view(batch_size, self.num_layers, self.hidden_dim)
        c0 = c0.permute(1, 0, 2).contiguous()

        # Zero teacher-forcing input
        decoder_input = torch.zeros(
            batch_size, seq_len, self.n_features,
            device=x.device, dtype=x.dtype
        )
        output, _ = self.decoder_lstm(decoder_input, (h0, c0))

        recon_seq = self.fc_out(output)  # [batch, seq_len, n_features]
        return recon_seq.transpose(1, 2)  # [batch, n_features, window_size]

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """MSE reconstruction loss only (no KL)."""
        loss = F.mse_loss(recon, x)
        return loss, {'recon_loss': loss}
