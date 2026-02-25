"""OmniAnomaly: Stochastic RNN with Normalizing Flows for Anomaly Detection.

Based on: Su et al. 2019 - "Robust Anomaly Detection for Multivariate Time
Series through Stochastic Recurrent Neural Network" (KDD 2019)

Reference: https://github.com/NetManAIOps/OmniAnomaly

Architecture:
- Stochastic RNN encoder with GRU cells
- Normalizing flows (planar flows) for flexible posterior
- GRU decoder from z_transformed (not hidden state)
- ELBO loss with learned output variance
- Reconstruction probability for anomaly scoring
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaselineModel


class PlanarFlow(nn.Module):
    """Planar normalizing flow: f(z) = z + u * h(w^T * z + b)."""

    def __init__(self, dim: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        linear = torch.sum(self.w * z, dim=-1, keepdim=True) + self.b
        h = torch.tanh(linear)
        h_prime = 1 - h ** 2

        z_out = z + self.u * h

        psi = h_prime * self.w
        log_det = torch.log(torch.abs(1 + torch.sum(psi * self.u, dim=-1)) + 1e-8)

        return z_out, log_det


class NormalizingFlows(nn.Module):
    """Stack of planar flows."""

    def __init__(self, dim: int, n_flows: int = 4):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sum_log_det = 0.0
        for flow in self.flows:
            z, log_det = flow(z)
            sum_log_det = sum_log_det + log_det
        return z, sum_log_det


class StochasticGRUEncoder(nn.Module):
    """Stochastic GRU encoder with latent variable at each timestep."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # GRU for hidden state: takes (x_t, z_t) -> h_t
        self.gru_cell = nn.GRUCell(input_dim + latent_dim, hidden_dim)

        # Posterior: q(z_t | x_t, h_{t-1})
        self.posterior_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # Prior: p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def forward(self, x_t, h_prev, use_prior=False):
        """Forward one timestep.

        Returns: (h_t, z_t, posterior_params, prior_params)
        """
        prior_params = self.prior_net(h_prev)
        prior_mu, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

        if use_prior:
            z_t = self._reparameterize(prior_mu, prior_logvar)
            posterior_params = prior_params
        else:
            posterior_input = torch.cat([x_t, h_prev], dim=-1)
            posterior_params = self.posterior_net(posterior_input)
            posterior_mu, posterior_logvar = torch.chunk(posterior_params, 2, dim=-1)
            z_t = self._reparameterize(posterior_mu, posterior_logvar)

        gru_input = torch.cat([x_t, z_t], dim=-1)
        h_t = self.gru_cell(gru_input, h_prev)

        return h_t, z_t, posterior_params, prior_params

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


class GRUDecoder(nn.Module):
    """GRU-based decoder that reconstructs from z_transformed.

    Bug fix #3: Uses GRU decoder instead of MLP.
    Outputs mean and log-variance for Gaussian reconstruction.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project z_transformed to decoder hidden state
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)

        # Decoder GRU: single step, takes z_transformed as input
        self.gru_cell = nn.GRUCell(latent_dim, hidden_dim)

        # Output distribution parameters (Bug fix #4: learned variance)
        self.output_mu = nn.Linear(hidden_dim, output_dim)
        self.output_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_transformed, h_enc=None):
        """Decode from z_transformed.

        Args:
            z_transformed: [batch, latent_dim] from normalizing flow output
            h_enc: Optional encoder hidden state for initialization

        Returns:
            (mu, logvar) each [batch, output_dim]
        """
        if h_enc is not None:
            h_dec = h_enc
        else:
            h_dec = torch.tanh(self.z_to_hidden(z_transformed))

        h_dec = self.gru_cell(z_transformed, h_dec)

        mu = self.output_mu(h_dec)
        logvar = self.output_logvar(h_dec)

        return mu, logvar


class OmniAnomaly(BaselineModel):
    """OmniAnomaly: Stochastic RNN with Normalizing Flows.

    Fixed implementation:
    - Decoder takes z_transformed (not h) [Bug fix #1, #2]
    - GRU-based decoder [Bug fix #3]
    - ELBO loss with learned output variance [Bug fix #4]
    - Pure reconstruction probability scoring [Bug fix #5]
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_flows: int = 4,
        beta: float = 1.0,
        n_measurement_vars: int = None,
    ):
        super().__init__(
            name="OmniAnomaly",
            n_features=n_features,
            window_size=window_size,
            n_measurement_vars=n_measurement_vars,
        )

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Stochastic GRU encoder
        self.stochastic_gru = StochasticGRUEncoder(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        # Normalizing flows for posterior
        self.flows = NormalizingFlows(latent_dim, n_flows)

        # GRU decoder: reconstructs from z_transformed (Bug fix #1, #3)
        self.decoder = GRUDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=n_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning reconstruction (point estimate).

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Reconstruction [batch, n_features, window_size]
        """
        batch_size = x.shape[0]
        x_seq = x.transpose(1, 2)  # [batch, seq_len, features]

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        reconstructions = []
        self._kl_losses = []

        for t in range(x_seq.shape[1]):
            x_t = x_seq[:, t, :]

            # Encode
            h, z, posterior_params, prior_params = self.stochastic_gru(x_t, h)

            # Apply normalizing flows (Bug fix #2: use z_transformed)
            z_transformed, log_det = self.flows(z)

            # KL divergence
            kl = self._compute_kl(posterior_params, prior_params, log_det)
            self._kl_losses.append(kl)

            # Decode from z_transformed (Bug fix #1)
            mu, _ = self.decoder(z_transformed, h)
            reconstructions.append(mu)

        recon_seq = torch.stack(reconstructions, dim=1)  # [batch, seq_len, features]
        recon = recon_seq.transpose(1, 2)  # [batch, features, seq_len]
        return recon

    def forward_with_distribution(self, x: torch.Tensor):
        """Forward pass returning full output distribution for ELBO loss.

        Returns:
            (recon_mus, recon_logvars, kl_losses) - lists of per-timestep values
        """
        batch_size = x.shape[0]
        x_seq = x.transpose(1, 2)

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        recon_mus = []
        recon_logvars = []
        kl_losses = []

        for t in range(x_seq.shape[1]):
            x_t = x_seq[:, t, :]

            h, z, posterior_params, prior_params = self.stochastic_gru(x_t, h)
            z_transformed, log_det = self.flows(z)

            kl = self._compute_kl(posterior_params, prior_params, log_det)
            kl_losses.append(kl)

            mu, logvar = self.decoder(z_transformed, h)
            recon_mus.append(mu)
            recon_logvars.append(logvar)

        return recon_mus, recon_logvars, kl_losses

    def _compute_kl(self, posterior_params, prior_params, log_det):
        """Compute KL divergence with flow adjustment."""
        post_mu, post_logvar = torch.chunk(posterior_params, 2, dim=-1)
        prior_mu, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

        kl = 0.5 * (
            prior_logvar - post_logvar +
            (torch.exp(post_logvar) + (post_mu - prior_mu) ** 2) /
            (torch.exp(prior_logvar) + 1e-8) - 1
        )

        kl = kl.sum(dim=-1).mean() - log_det.mean()
        return kl

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute ELBO loss with learned output variance (Bug fix #4).

        ELBO = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
        where log p(x|z) = -0.5 * (logvar + (x-mu)^2 / exp(logvar))
        """
        x_seq = x.transpose(1, 2)  # [batch, seq_len, features]

        recon_mus, recon_logvars, kl_losses = self.forward_with_distribution(x)

        # Gaussian NLL reconstruction loss (Bug fix #4: ELBO with learned variance)
        # Loss on measurement channels only
        n_m = self.n_measurement_vars
        nll_total = 0.0
        for t in range(x_seq.shape[1]):
            x_t = x_seq[:, t, :n_m]
            mu_t = recon_mus[t][:, :n_m]
            logvar_t = recon_logvars[t][:, :n_m]

            # Negative log-likelihood of Gaussian
            nll_t = 0.5 * (logvar_t + (x_t - mu_t) ** 2 / (torch.exp(logvar_t) + 1e-8))
            nll_total = nll_total + nll_t.sum(dim=-1).mean()

        recon_loss = nll_total / x_seq.shape[1]

        # KL loss
        kl_loss = sum(kl_losses) / len(kl_losses)

        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
        }

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores using reconstruction probability (Bug fix #5).

        Score = negative log-likelihood under learned output distribution.
        Only scored on measurement channels.
        """
        x_seq = x.transpose(1, 2)  # [batch, seq_len, features]

        recon_mus, recon_logvars, _ = self.forward_with_distribution(x)

        n_m = self.n_measurement_vars

        # Sum negative log-prob across time and measurement channels
        total_nll = torch.zeros(x.shape[0], device=x.device)
        for t in range(x_seq.shape[1]):
            x_t = x_seq[:, t, :n_m]  # Only measurement channels
            mu_t = recon_mus[t][:, :n_m]
            logvar_t = recon_logvars[t][:, :n_m]

            nll_t = 0.5 * (logvar_t + (x_t - mu_t) ** 2 / (torch.exp(logvar_t) + 1e-8))
            total_nll = total_nll + nll_t.sum(dim=-1)

        # Average over time
        anomaly_score = total_nll / x_seq.shape[1]

        return anomaly_score

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'beta': self.beta,
        })
        return info
