"""LSTM: Plain LSTM Encoder-Decoder for Anomaly Detection.

Simple LSTM reconstruction baseline without VAE components.

Architecture:
- Encoder: Bidirectional LSTM -> concat fwd+bwd -> project to hidden_dim
- Decoder: Unidirectional LSTM, init from encoder hidden, zero input
- Loss: MSE reconstruction only (no KL divergence)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaselineModel


class LSTM(BaselineModel):
    """Plain LSTM encoder-decoder for anomaly detection."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_measurement_vars: int = None,
    ):
        super().__init__(
            name="LSTM",
            n_features=n_features,
            window_size=window_size,
            n_measurement_vars=n_measurement_vars,
        )

        self.hidden_dim = hidden_dim
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

        # Project bidirectional output to hidden_dim for decoder init
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)

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
        # Transpose to [batch, seq_len, n_features]
        x_seq = x.transpose(1, 2)

        # Encode
        _, (h_n, c_n) = self.encoder_lstm(x_seq)

        # h_n: [num_layers*2, batch, hidden_dim] for bidirectional
        # Concatenate fwd/bwd for each layer, project to decoder hidden_dim
        h_fwd = h_n[0::2]  # [num_layers, batch, hidden_dim]
        h_bwd = h_n[1::2]  # [num_layers, batch, hidden_dim]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # [num_layers, batch, hidden_dim*2]
        h0 = self.fc_h(h_cat)  # [num_layers, batch, hidden_dim]

        c_fwd = c_n[0::2]
        c_bwd = c_n[1::2]
        c_cat = torch.cat([c_fwd, c_bwd], dim=-1)
        c0 = self.fc_c(c_cat)

        # Decode with zero input (rely on hidden state)
        decoder_input = torch.zeros(
            batch_size, seq_len, self.n_features,
            device=x.device, dtype=x.dtype
        )
        output, _ = self.decoder_lstm(decoder_input, (h0, c0))

        # Project to output
        recon_seq = self.fc_out(output)  # [batch, seq_len, n_features]
        return recon_seq.transpose(1, 2)  # [batch, n_features, window_size]

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """MSE reconstruction loss."""
        loss = F.mse_loss(recon, x)
        return loss, {'recon_loss': loss}
