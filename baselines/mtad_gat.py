"""MTAD-GAT: Multivariate Time-series Anomaly Detection via Graph Attention Network.

Based on: Zhao et al. 2020 - "Multivariate Time-series Anomaly Detection via
Graph Attention Network"

Architecture:
- Dual-branch: Temporal convolution + Graph attention
- Multi-scale temporal modeling
- Joint reconstruction + forecasting
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaselineModel


class TemporalConvBlock(nn.Module):
    """Temporal convolution block with dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [batch, channels, seq_len]

        Returns:
            Output [batch, out_channels, seq_len]
        """
        residual = self.residual(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal encoder using dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            TemporalConvBlock(in_channels, hidden_dim, kernel_size, 1, dropout)
        )

        # Subsequent layers with increasing dilation
        for i in range(1, n_layers):
            dilation = 2 ** i
            self.layers.append(
                TemporalConvBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [batch, channels, seq_len]

        Returns:
            Output [batch, hidden_dim, seq_len]
        """
        for layer in self.layers:
            x = layer(x)
        return x


class FeatureGraphAttention(nn.Module):
    """Graph attention over feature (sensor) dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads

        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        self.W_o = nn.Linear(out_features, out_features)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention across features.

        Args:
            x: Input [batch, n_features, hidden_dim]

        Returns:
            Output [batch, n_features, out_features]
        """
        batch_size, n_features, _ = x.shape

        # Linear projections
        Q = self.W_q(x).view(batch_size, n_features, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, n_features, self.n_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, n_features, self.n_heads, self.head_dim)

        # Transpose for attention: [batch, heads, n_features, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.matmul(attn, V)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, n_features, -1)
        out = self.W_o(out)

        return out


class TemporalGraphAttention(nn.Module):
    """Graph attention over temporal dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads

        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        self.W_o = nn.Linear(out_features, out_features)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention across time.

        Args:
            x: Input [batch, seq_len, hidden_dim]

        Returns:
            Output [batch, seq_len, out_features]
        """
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.W_o(out)

        return out


class MTADGAT(BaselineModel):
    """MTAD-GAT: Multi-scale Temporal Anomaly Detection with Graph Attention.

    This model combines:
    - Multi-scale temporal convolutions for local pattern extraction
    - Feature-wise graph attention for sensor correlations
    - Temporal graph attention for temporal dependencies
    - Joint reconstruction and forecasting objectives
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
    ):
        """Initialize MTAD-GAT model.

        Args:
            n_features: Number of input features/sensors
            window_size: Temporal window size
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            n_temporal_layers: Number of temporal conv layers
            dropout: Dropout rate
            forecast_horizon: Number of steps to forecast
        """
        super().__init__(
            name="MTAD-GAT",
            n_features=n_features,
            window_size=window_size,
        )

        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        # Multi-scale temporal encoder
        self.temporal_encoder = MultiScaleTemporalEncoder(
            in_channels=n_features,
            hidden_dim=hidden_dim,
            n_layers=n_temporal_layers,
            dropout=dropout,
        )

        # Feature graph attention (across sensors)
        self.feature_gat = FeatureGraphAttention(
            in_features=hidden_dim,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Temporal graph attention
        self.temporal_gat = TemporalGraphAttention(
            in_features=hidden_dim,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Reconstruction decoder
        self.recon_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features),
        )

        # Forecasting decoder
        self.forecast_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features * forecast_horizon),
        )

        # Loss weights
        self.recon_weight = 0.5
        self.forecast_weight = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning reconstruction.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Reconstruction [batch, n_features, window_size]
        """
        recon, _ = self.forward_full(x)
        return recon

    def forward_full(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass with reconstruction and forecast.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Tuple of (reconstruction, forecast)
        """
        batch_size = x.shape[0]

        # Temporal encoding: [batch, hidden, window]
        temporal_features = self.temporal_encoder(x)

        # Feature attention: [batch, window, hidden] -> attend over features
        # First pool temporal for feature attention
        feature_input = temporal_features.mean(dim=-1).unsqueeze(1)  # [batch, 1, hidden]
        feature_input = feature_input.expand(-1, self.n_features, -1)

        # Create feature-level representation
        feature_repr = temporal_features.transpose(1, 2)  # [batch, window, hidden]

        # Feature graph attention
        feature_attn = self.feature_gat(
            feature_repr.mean(dim=1, keepdim=True).expand(-1, self.n_features, -1)
        )

        # Temporal graph attention
        temporal_attn = self.temporal_gat(feature_repr)

        # Pool temporal attention
        temporal_pooled = temporal_attn.mean(dim=1)  # [batch, hidden]
        feature_pooled = feature_attn.mean(dim=1)  # [batch, hidden]

        # Fuse
        fused = self.fusion(torch.cat([temporal_pooled, feature_pooled], dim=-1))

        # Expand for per-timestep reconstruction
        fused_expanded = fused.unsqueeze(1).expand(-1, self.window_size, -1)

        # Decode reconstruction
        recon = self.recon_decoder(temporal_attn)  # [batch, window, n_features]
        recon = recon.transpose(1, 2)  # [batch, n_features, window]

        # Decode forecast
        forecast = self.forecast_decoder(fused)  # [batch, n_features * horizon]
        forecast = forecast.view(batch_size, self.n_features, self.forecast_horizon)

        return recon, forecast

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute joint reconstruction and forecasting loss.

        Args:
            x: Original input [batch, n_features, window_size]
            recon: Reconstruction

        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Get full outputs
        recon, forecast = self.forward_full(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # Forecasting loss (predict last timesteps from earlier context)
        # Use last `forecast_horizon` timesteps as target
        forecast_target = x[:, :, -self.forecast_horizon:]
        forecast_loss = F.mse_loss(forecast, forecast_target, reduction='mean')

        # Combined loss
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
        """Compute anomaly scores combining reconstruction and forecast errors.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Anomaly scores [batch]
        """
        recon, forecast = self.forward_full(x)

        # Reconstruction error
        recon_error = torch.mean((x - recon) ** 2, dim=(1, 2))

        # Forecast error
        forecast_target = x[:, :, -self.forecast_horizon:]
        forecast_error = torch.mean((forecast_target - forecast) ** 2, dim=(1, 2))

        # Combined score
        anomaly_score = (
            self.recon_weight * recon_error +
            self.forecast_weight * forecast_error
        )

        return anomaly_score

    def get_model_info(self) -> Dict:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            'hidden_dim': self.hidden_dim,
            'forecast_horizon': self.forecast_horizon,
        })
        return info
