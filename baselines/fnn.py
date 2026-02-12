"""FNN: Feedforward Neural Network for Anomaly Detection.

Simple feedforward reconstruction network (no bottleneck, unlike AE).

Architecture:
- Flatten [B, N, W] -> [B, N*W]
- Network: N*W -> 256 -> ReLU -> Dropout -> 128 -> ReLU -> Dropout ->
           256 -> ReLU -> Dropout -> N*W
- Loss: MSE reconstruction
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaselineModel


class FNN(BaselineModel):
    """Feedforward Neural Network for multivariate time series anomaly detection."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        n_measurement_vars: int = None,
    ):
        super().__init__(
            name="FNN",
            n_features=n_features,
            window_size=window_size,
            n_measurement_vars=n_measurement_vars,
        )

        if hidden_dims is None:
            hidden_dims = [256, 128, 256]

        self.hidden_dims = hidden_dims
        input_dim = n_features * window_size

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Reconstruction [batch, n_features, window_size]
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        recon = self.network(x_flat)
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
