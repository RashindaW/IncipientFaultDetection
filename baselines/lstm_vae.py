"""LSTM-VAE: LSTM Variational Autoencoder for Anomaly Detection.

Based on: Park et al. 2018 - "A Multimodal Anomaly Detector for Robot-Assisted
Feeding Using an LSTM-Based Variational Autoencoder"

Architecture:
- Bidirectional LSTM encoder -> latent μ, log(σ²)
- Reparameterization: z = μ + σ * ε
- LSTM decoder with z as initial hidden state
- Loss: Reconstruction MSE + β * KL(q(z|x) || p(z))
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaselineModel


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder that outputs latent distribution parameters."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project bidirectional hidden states to latent parameters
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent distribution parameters.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Tuple of (mu, logvar) each of shape [batch, latent_dim]
        """
        # LSTM encoding
        _, (h_n, _) = self.lstm(x)

        # Concatenate forward and backward hidden states from last layer
        h_forward = h_n[-2]  # [batch, hidden_dim]
        h_backward = h_n[-1]  # [batch, hidden_dim]
        h_combined = torch.cat([h_forward, h_backward], dim=-1)

        # Project to latent parameters
        mu = self.fc_mu(h_combined)
        logvar = self.fc_logvar(h_combined)

        return mu, logvar


class LSTMDecoder(nn.Module):
    """LSTM decoder that reconstructs sequence from latent representation."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project latent to initial hidden state
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        # Decoder LSTM
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """Decode latent representation to output sequence.

        Args:
            z: Latent representation [batch, latent_dim]
            seq_len: Output sequence length (default: self.seq_len)

        Returns:
            Reconstructed sequence [batch, seq_len, output_dim]
        """
        batch_size = z.shape[0]
        if seq_len is None:
            seq_len = self.seq_len

        # Initialize hidden states from latent
        h0 = self.fc_hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
        c0 = self.fc_cell(z).view(self.num_layers, batch_size, self.hidden_dim)

        # Create input sequence (start with zeros, autoregressively not used here)
        # We use teacher forcing style: input zeros and rely on hidden state
        decoder_input = torch.zeros(
            batch_size, seq_len, self.fc_out.out_features,
            device=z.device, dtype=z.dtype
        )

        # Decode
        output, _ = self.lstm(decoder_input, (h0, c0))

        # Project to output dimension
        recon = self.fc_out(output)

        return recon


class LSTMVAE(BaselineModel):
    """LSTM Variational Autoencoder for multivariate time series anomaly detection.

    This model uses:
    - Bidirectional LSTM encoder to capture temporal patterns
    - Variational latent space with reparameterization trick
    - LSTM decoder for sequence reconstruction
    - Combined reconstruction + KL divergence loss

    Anomaly score is computed as reconstruction probability (negative log-likelihood).
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        beta: float = 1.0,
        n_measurement_vars: int = None,
    ):
        """Initialize LSTM-VAE model.

        Args:
            n_features: Number of input features/channels
            window_size: Temporal window size
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent space dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            beta: Weight for KL divergence term (β-VAE)
            n_measurement_vars: Number of measurement vars for scoring
        """
        super().__init__(
            name="LSTM-VAE",
            n_features=n_features,
            window_size=window_size,
            n_measurement_vars=n_measurement_vars,
        )

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = LSTMEncoder(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Decoder
        self.decoder = LSTMDecoder(
            output_dim=n_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            seq_len=window_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log variance of latent distribution [batch, latent_dim]

        Returns:
            Sampled latent vector [batch, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use mean
            return mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor [batch, n_features, window_size]

        Returns:
            Reconstructed tensor [batch, n_features, window_size]
        """
        # Transpose to [batch, seq_len, features] for LSTM
        x_seq = x.transpose(1, 2)  # [batch, window_size, n_features]

        # Encode
        mu, logvar = self.encoder(x_seq)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_seq = self.decoder(z, seq_len=x_seq.shape[1])

        # Transpose back to [batch, n_features, window_size]
        recon = recon_seq.transpose(1, 2)

        # Store latent parameters for loss computation
        self._mu = mu
        self._logvar = logvar

        return recon

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor [batch, n_features, window_size]

        Returns:
            Tuple of (mu, logvar)
        """
        x_seq = x.transpose(1, 2)
        return self.encoder(x_seq)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation.

        Args:
            z: Latent vector [batch, latent_dim]

        Returns:
            Reconstructed tensor [batch, n_features, window_size]
        """
        recon_seq = self.decoder(z)
        return recon_seq.transpose(1, 2)

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Original input [batch, n_features, window_size]
            recon: Reconstructed output [batch, n_features, window_size]

        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Reconstruction loss (MSE) on measurement channels only
        n_m = self.n_measurement_vars
        recon_loss = F.mse_loss(recon[:, :n_m], x[:, :n_m], reduction='mean')

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(
            1 + self._logvar - self._mu.pow(2) - self._logvar.exp()
        )

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss

        loss_components = {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
        }

        return total_loss, loss_components

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores based on reconstruction probability.

        Scores only on measurement channels (excludes control variables).

        Args:
            x: Input tensor [batch, n_features, window_size]

        Returns:
            Anomaly scores [batch]
        """
        # Forward pass
        recon = self.forward(x)

        # Reconstruction error per sample (measurement channels only)
        n_m = self.n_measurement_vars
        recon_error = torch.mean((x[:, :n_m] - recon[:, :n_m]) ** 2, dim=(1, 2))

        # KL divergence per sample
        kl_per_sample = -0.5 * torch.sum(
            1 + self._logvar - self._mu.pow(2) - self._logvar.exp(),
            dim=-1
        )

        # Combined anomaly score (negative log-likelihood proxy)
        anomaly_score = recon_error + self.beta * kl_per_sample

        return anomaly_score

    def get_model_info(self) -> Dict:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'beta': self.beta,
        })
        return info
