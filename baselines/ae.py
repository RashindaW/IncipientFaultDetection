"""AE: MLP Autoencoder for Anomaly Detection.

Simple autoencoder baseline that flattens the multivariate time series window
and reconstructs it through an MLP bottleneck.

Architecture:
- Flatten [B, N, W] -> [B, N*W]
- Encoder: N*W -> 20 -> 20 -> 10 -> 10 -> 4 (ReLU + BN between layers)
- Decoder: Mirror 4 -> 10 -> 10 -> 20 -> 20 -> N*W
- Loss: MSE reconstruction
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaselineModel


class AE(BaselineModel):
    """MLP Autoencoder for multivariate time series anomaly detection."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 4,
        n_measurement_vars: int = None,
    ):
        super().__init__(
            name="AE",
            n_features=n_features,
            window_size=window_size,
            n_measurement_vars=n_measurement_vars,
        )

        if hidden_dims is None:
            hidden_dims = [20, 20, 10, 10]

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        input_dim = n_features * window_size

        # Encoder
        enc_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
            ])
            prev_dim = h_dim
        enc_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
            ])
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Reconstruction [batch, n_features, window_size]
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        z = self.encoder(x_flat)
        recon = self.decoder(z)
        return recon.view(batch_size, self.n_features, self.window_size)

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """MSE reconstruction loss."""
        loss = F.mse_loss(recon, x)
        return loss, {'recon_loss': loss}
