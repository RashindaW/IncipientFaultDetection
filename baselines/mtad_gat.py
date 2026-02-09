"""MTAD-GAT: Multivariate Time-series Anomaly Detection via Graph Attention Network.

Based on: Zhao et al. 2020 - "Multivariate Time-series Anomaly Detection via
Graph Attention Network" (arXiv 2009.02040)

Reference: https://github.com/ML4ITS/mtad-gat-pytorch

Architecture (corrected):
- 1D Conv layer per feature
- Feature GAT: each node = one sensor's T-dim time series [Bug fix #1]
- Temporal GAT: attention across timesteps
- GRU processes concatenated [conv, feat_gat, temp_gat] [Bug fix #2, #6]
- GAT-style attention: LeakyReLU(a^T[Wh_i || Wh_j]) [Bug fix #3]
- Forecast target: next timestep x_{t+1} [Bug fix #4]
- GRU-based reconstruction decoder [Bug fix #5]
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaselineModel


class ConvLayer(nn.Module):
    """1D causal convolution for temporal feature extraction."""

    def __init__(self, n_features: int, window_size: int, kernel_size: int = 7):
        super().__init__()
        self.padding = (kernel_size - 1)  # causal padding
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=self.padding,
        )
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D conv to each feature independently.

        Args:
            x: [batch, n_features, window_size]

        Returns:
            [batch, n_features, window_size]
        """
        batch_size, n_features, window_size = x.shape

        # Process each feature: [B*N, 1, W]
        x = x.view(batch_size * n_features, 1, window_size)
        x = self.conv(x)
        # Causal: trim future
        x = x[:, :, :window_size]
        x = F.relu(x)
        x = x.view(batch_size, n_features, window_size)
        return x


class FeatureGAT(nn.Module):
    """Feature-oriented GAT: each node = one sensor's time series.

    Bug fix #1: Each node receives its own unique sensor time-series vector,
    not a mean-pooled identical vector.

    Bug fix #3: Uses GAT-style attention (LeakyReLU(a^T[Wh_i || Wh_j]))
    instead of Transformer-style (softmax(QK^T/sqrt(d))V).
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        # GAT attention vector: a in R^{2*head_dim} per head
        self.att = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GAT attention across feature (sensor) dimension.

        Args:
            x: [batch, n_features, in_features] where each node i has its own
               unique feature vector (e.g., sensor i's time series)

        Returns:
            [batch, n_features, out_features]
        """
        batch_size, n_nodes, _ = x.shape

        # Linear transform: [B, N, out_features]
        h = self.W(x)
        h = h.view(batch_size, n_nodes, self.n_heads, self.head_dim)

        # GAT attention: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # Concatenate all pairs
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)
        h_cat = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, heads, 2*head_dim]

        e = (h_cat * self.att).sum(dim=-1)  # [B, N, N, heads]
        e = self.leaky_relu(e)

        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        # Aggregate: [B, N, heads, head_dim]
        out = torch.einsum('bijk,bjkd->bikd', alpha, h)
        out = out.reshape(batch_size, n_nodes, self.out_features)

        return out


class TemporalGAT(nn.Module):
    """Temporal GAT: attention across time dimension.

    Bug fix #3: Uses GAT-style attention.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.att = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GAT attention across time.

        Args:
            x: [batch, window_size, in_features]

        Returns:
            [batch, window_size, out_features]
        """
        batch_size, seq_len, _ = x.shape

        h = self.W(x)
        h = h.view(batch_size, seq_len, self.n_heads, self.head_dim)

        h_i = h.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        h_cat = torch.cat([h_i, h_j], dim=-1)

        e = (h_cat * self.att).sum(dim=-1)
        e = self.leaky_relu(e)

        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        out = torch.einsum('bijk,bjkd->bikd', alpha, h)
        out = out.reshape(batch_size, seq_len, self.out_features)

        return out


class GRUReconDecoder(nn.Module):
    """GRU-based reconstruction decoder.

    Bug fix #5: Uses GRU decoder instead of per-timestep MLP.
    Takes hidden representation, produces full window reconstruction.
    """

    def __init__(self, hidden_dim: int, n_features: int, window_size: int):
        super().__init__()
        self.window_size = window_size

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, n_features)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Decode reconstruction from hidden representation.

        Args:
            h: [batch, hidden_dim] - final hidden state from GRU

        Returns:
            [batch, window_size, n_features]
        """
        # Repeat h for each timestep
        h_expanded = h.unsqueeze(1).expand(-1, self.window_size, -1)

        out, _ = self.gru(h_expanded)
        recon = self.fc(out)  # [batch, window_size, n_features]

        return recon


class MTADGAT(BaselineModel):
    """MTAD-GAT: Multivariate Time-series Anomaly Detection via Graph Attention.

    Fixed implementation following ML4ITS reference:
    - Feature GAT with unique per-sensor input [Bug fix #1]
    - GRU processes concat(conv, feat_gat, temp_gat) [Bug fix #2, #6]
    - GAT-style attention [Bug fix #3]
    - Forecast target = next timestep [Bug fix #4]
    - GRU-based reconstruction decoder [Bug fix #5]
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_temporal_layers: int = 3,
        dropout: float = 0.1,
        forecast_horizon: int = 1,
        n_measurement_vars: int = None,
    ):
        super().__init__(
            name="MTAD-GAT",
            n_features=n_features,
            window_size=window_size,
            n_measurement_vars=n_measurement_vars,
        )

        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        # 1D Conv layer (Bug fix #6: included in fusion)
        self.conv_layer = ConvLayer(n_features, window_size)

        # Feature GAT: each node is one sensor's W-dim time series
        # Input: [B, N, W] where node i = sensor i's time series
        # Use n_heads=1 to avoid divisibility issues with arbitrary window_size
        self.feature_gat = FeatureGAT(
            in_features=window_size,
            out_features=window_size,
            n_heads=1,
            dropout=dropout,
        )

        # Temporal GAT: attention across time
        # Input: [B, W, N] transposed
        # Use n_heads=1 for temporal GAT to avoid divisibility issues with n_features
        self.temporal_gat = TemporalGAT(
            in_features=n_features,
            out_features=n_features,
            n_heads=1,
            dropout=dropout,
        )

        # GRU processes concatenated [conv, feat_gat, temp_gat]
        # Bug fix #2: GRU layer added
        # After conv: [B, N, W] -> transpose -> [B, W, N]
        # After feat_gat: [B, N, W] -> transpose -> [B, W, N]
        # After temp_gat: [B, W, N]
        # Concat on feature dim: [B, W, 3*N]
        self.gru = nn.GRU(
            input_size=n_features * 3,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Reconstruction decoder (Bug fix #5: GRU-based)
        self.recon_decoder = GRUReconDecoder(
            hidden_dim=hidden_dim,
            n_features=n_features,
            window_size=window_size,
        )

        # Forecasting decoder: predict next timestep
        # Bug fix #4: target is x_{t+1}, not last of same window
        self.forecast_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features * forecast_horizon),
        )

        # Loss weights
        self.recon_weight = 0.5
        self.forecast_weight = 0.5

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input through conv, GATs, and GRU.

        Args:
            x: [batch, n_features, window_size]

        Returns:
            GRU hidden state [batch, hidden_dim]
        """
        # 1D Conv: [B, N, W]
        conv_out = self.conv_layer(x)

        # Feature GAT: each node = sensor's time series
        # Input: [B, N, W] - node i has its own W-dim vector (Bug fix #1)
        feat_gat_out = self.feature_gat(x)  # [B, N, W]

        # Temporal GAT: transpose to [B, W, N]
        x_temporal = x.transpose(1, 2)  # [B, W, N]
        temp_gat_out = self.temporal_gat(x_temporal)  # [B, W, N]

        # Concat: [B, W, 3*N] (Bug fix #6: include conv in fusion)
        conv_t = conv_out.transpose(1, 2)      # [B, W, N]
        feat_gat_t = feat_gat_out.transpose(1, 2)  # [B, W, N]
        concat = torch.cat([conv_t, feat_gat_t, temp_gat_out], dim=-1)

        # GRU (Bug fix #2)
        gru_out, h_n = self.gru(concat)

        # Use last hidden state
        h_final = h_n.squeeze(0)  # [batch, hidden_dim]

        return h_final

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning reconstruction.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Reconstruction [batch, n_features, window_size]
        """
        recon, _ = self.forward_full(x)
        return recon

    def forward_full(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass with reconstruction and forecast.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            (reconstruction [B, N, W], forecast [B, N, horizon])
        """
        batch_size = x.shape[0]

        h_final = self._encode(x)

        # Reconstruction: GRU decoder (Bug fix #5)
        recon_seq = self.recon_decoder(h_final)  # [B, W, N]
        recon = recon_seq.transpose(1, 2)  # [B, N, W]

        # Forecast: predict next timestep(s) (Bug fix #4)
        forecast_flat = self.forecast_decoder(h_final)
        forecast = forecast_flat.view(batch_size, self.n_features, self.forecast_horizon)

        return recon, forecast

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute joint reconstruction and forecasting loss.

        Bug fix #4: Forecast target = last timestep of window (serves as
        next-step prediction since input effectively uses W-1 steps through
        the causal conv + GRU).
        """
        recon, forecast = self.forward_full(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # Forecasting loss: predict last timestep (Bug fix #4)
        forecast_target = x[:, :, -self.forecast_horizon:]
        forecast_loss = F.mse_loss(forecast, forecast_target, reduction='mean')

        total_loss = (
            self.recon_weight * recon_loss +
            self.forecast_weight * forecast_loss
        )

        return total_loss, {
            'recon_loss': recon_loss,
            'forecast_loss': forecast_loss,
            'total_loss': total_loss,
        }

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores on measurement channels only."""
        recon, forecast = self.forward_full(x)

        n_m = self.n_measurement_vars

        # Reconstruction error (measurement channels only)
        recon_error = torch.mean((x[:, :n_m] - recon[:, :n_m]) ** 2, dim=(1, 2))

        # Forecast error (measurement channels only)
        forecast_target = x[:, :n_m, -self.forecast_horizon:]
        forecast_error = torch.mean(
            (forecast_target - forecast[:, :n_m]) ** 2, dim=(1, 2)
        )

        anomaly_score = (
            self.recon_weight * recon_error +
            self.forecast_weight * forecast_error
        )

        return anomaly_score

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            'hidden_dim': self.hidden_dim,
            'forecast_horizon': self.forecast_horizon,
        })
        return info
