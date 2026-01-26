"""OmniAnomaly: Stochastic RNN with Normalizing Flows for Anomaly Detection.

Based on: Su et al. 2019 - "Robust Anomaly Detection for Multivariate Time
Series through Stochastic Recurrent Neural Network"

Architecture:
- Stochastic RNN with GRU cells
- Normalizing flows (planar flows) for flexible latent distribution
- Reconstruction probability for anomaly scoring
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaselineModel


class PlanarFlow(nn.Module):
    """Planar normalizing flow transformation.

    f(z) = z + u * h(w^T * z + b)

    This allows learning flexible posterior distributions.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply planar flow transformation.

        Args:
            z: Input latent [batch, dim]

        Returns:
            Tuple of (transformed_z, log_det_jacobian)
        """
        # Activation: tanh
        linear = torch.sum(self.w * z, dim=-1, keepdim=True) + self.b
        h = torch.tanh(linear)
        h_prime = 1 - h ** 2  # tanh derivative

        # Transform
        z_out = z + self.u * h

        # Log determinant of Jacobian
        psi = h_prime * self.w
        log_det = torch.log(torch.abs(1 + torch.sum(psi * self.u, dim=-1)) + 1e-8)

        return z_out, log_det


class NormalizingFlows(nn.Module):
    """Stack of planar flows for flexible posterior."""

    def __init__(self, dim: int, n_flows: int = 4):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply sequence of flows.

        Args:
            z: Input latent [batch, dim]

        Returns:
            Tuple of (transformed_z, sum_log_det)
        """
        sum_log_det = 0.0
        for flow in self.flows:
            z, log_det = flow(z)
            sum_log_det = sum_log_det + log_det
        return z, sum_log_det


class StochasticGRU(nn.Module):
    """Stochastic GRU cell with latent variable at each timestep."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # GRU for hidden state
        self.gru_cell = nn.GRUCell(input_dim + latent_dim, hidden_dim)

        # Posterior network: q(z_t | x_t, h_{t-1})
        self.posterior_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mu and logvar
        )

        # Prior network: p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        use_prior: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward one timestep.

        Args:
            x_t: Input at time t [batch, input_dim]
            h_prev: Previous hidden state [batch, hidden_dim]
            use_prior: If True, sample from prior (for generation)

        Returns:
            Tuple of (h_t, z_t, posterior_params, prior_params)
        """
        # Prior: p(z_t | h_{t-1})
        prior_params = self.prior_net(h_prev)
        prior_mu, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

        if use_prior:
            # Sample from prior
            z_t = self._reparameterize(prior_mu, prior_logvar)
            posterior_params = prior_params
        else:
            # Posterior: q(z_t | x_t, h_{t-1})
            posterior_input = torch.cat([x_t, h_prev], dim=-1)
            posterior_params = self.posterior_net(posterior_input)
            posterior_mu, posterior_logvar = torch.chunk(posterior_params, 2, dim=-1)
            z_t = self._reparameterize(posterior_mu, posterior_logvar)

        # Update hidden state
        gru_input = torch.cat([x_t, z_t], dim=-1)
        h_t = self.gru_cell(gru_input, h_prev)

        return h_t, z_t, posterior_params, prior_params

    def _reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


class OmniAnomaly(BaselineModel):
    """OmniAnomaly: Stochastic RNN with Normalizing Flows.

    This model combines:
    - Stochastic RNN for temporal modeling with latent variables
    - Normalizing flows for flexible posterior distributions
    - Reconstruction probability for anomaly scoring
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_flows: int = 4,
        beta: float = 1.0,
    ):
        """Initialize OmniAnomaly model.

        Args:
            n_features: Number of input features
            window_size: Temporal window size
            hidden_dim: GRU hidden dimension
            latent_dim: Latent variable dimension
            n_flows: Number of normalizing flow transformations
            beta: Weight for KL divergence term
        """
        super().__init__(
            name="OmniAnomaly",
            n_features=n_features,
            window_size=window_size,
        )

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Stochastic GRU
        self.stochastic_gru = StochasticGRU(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        # Normalizing flows for posterior
        self.flows = NormalizingFlows(latent_dim, n_flows)

        # Decoder: reconstruct from hidden state
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

        # Output distribution parameters (for reconstruction probability)
        self.output_mu = nn.Linear(hidden_dim, n_features)
        self.output_logvar = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning reconstruction.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Reconstruction [batch, n_features, window_size]
        """
        batch_size = x.shape[0]

        # Transpose to [batch, seq_len, features]
        x_seq = x.transpose(1, 2)

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        reconstructions = []
        self._kl_losses = []

        for t in range(x_seq.shape[1]):
            x_t = x_seq[:, t, :]

            # Stochastic GRU step
            h, z, posterior_params, prior_params = self.stochastic_gru(x_t, h)

            # Apply normalizing flows
            z_transformed, log_det = self.flows(z)

            # Compute KL divergence
            kl = self._compute_kl(posterior_params, prior_params, log_det)
            self._kl_losses.append(kl)

            # Decode
            recon_t = self.decoder(h)
            reconstructions.append(recon_t)

        # Stack reconstructions [batch, seq_len, features]
        recon_seq = torch.stack(reconstructions, dim=1)

        # Transpose back to [batch, features, seq_len]
        recon = recon_seq.transpose(1, 2)

        return recon

    def forward_with_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with reconstruction probability.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Tuple of (reconstruction, reconstruction_probability)
        """
        batch_size = x.shape[0]
        x_seq = x.transpose(1, 2)

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        recon_probs = []

        for t in range(x_seq.shape[1]):
            x_t = x_seq[:, t, :]
            h, z, _, _ = self.stochastic_gru(x_t, h)

            # Output distribution
            mu = self.output_mu(h)
            logvar = self.output_logvar(h)

            # Log probability of x_t under Gaussian
            log_prob = -0.5 * (
                logvar + (x_t - mu) ** 2 / (torch.exp(logvar) + 1e-8)
            )
            recon_probs.append(log_prob.sum(dim=-1))

        # Sum of log probs across time [batch]
        recon_prob = torch.stack(recon_probs, dim=1).sum(dim=1)

        # Also get reconstruction
        recon = self.forward(x)

        return recon, recon_prob

    def _compute_kl(
        self,
        posterior_params: torch.Tensor,
        prior_params: torch.Tensor,
        log_det: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence with flow adjustment.

        Args:
            posterior_params: [mu, logvar] from posterior
            prior_params: [mu, logvar] from prior
            log_det: Log determinant from normalizing flows

        Returns:
            KL divergence scalar
        """
        post_mu, post_logvar = torch.chunk(posterior_params, 2, dim=-1)
        prior_mu, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

        # Standard Gaussian KL
        kl = 0.5 * (
            prior_logvar - post_logvar +
            (torch.exp(post_logvar) + (post_mu - prior_mu) ** 2) /
            (torch.exp(prior_logvar) + 1e-8) - 1
        )

        # Adjust for normalizing flow
        kl = kl.sum(dim=-1).mean() - log_det.mean()

        return kl

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute OmniAnomaly loss.

        Args:
            x: Original input
            recon: Reconstruction

        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # KL loss (summed over timesteps)
        if hasattr(self, '_kl_losses') and self._kl_losses:
            kl_loss = sum(self._kl_losses) / len(self._kl_losses)
        else:
            kl_loss = torch.tensor(0.0, device=x.device)

        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
        }

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores using reconstruction probability.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Anomaly scores [batch] (higher = more anomalous)
        """
        recon, recon_prob = self.forward_with_prob(x)

        # Reconstruction error (MSE per sample)
        mse = torch.mean((x - recon) ** 2, dim=(1, 2))

        # Anomaly score: negative reconstruction probability + MSE
        # (lower prob = higher anomaly)
        anomaly_score = mse - 0.1 * recon_prob

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
