"""USAD: UnSupervised Anomaly Detection on Multivariate Time Series.

Based on: Audibert et al. 2020 - "USAD: UnSupervised Anomaly Detection on
Multivariate Time Series" (KDD 2020)

Reference: https://github.com/manigalati/usad

Architecture:
- Shared encoder E
- Two decoders G1, G2
- Two optimizers: opt1 for (E, G1), opt2 for (E, G2)
- L_AE1 = (1/n)||W - AE1(W)|| + (1-1/n)||W - AE2(AE1(W))||
- L_AE2 = (1/n)||W - AE2(W)|| - (1-1/n)||W - AE2(AE1(W))||
- Score: α * ||x - AE1(x)|| + β * ||x - AE2(AE1(x))||
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaselineModel


class USADEncoder(nn.Module):
    """Shared encoder for USAD."""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class USADDecoder(nn.Module):
    """Decoder for USAD (G1 or G2)."""

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()

        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class USAD(BaselineModel):
    """USAD: UnSupervised Anomaly Detection.

    Two-optimizer adversarial training between:
    - AE1 = G1(E(W)): standard autoencoder
    - AE2 = G2(E(W)): second decoder that also discriminates
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dims: list = None,
        latent_dim: int = 32,
        alpha: float = 1.0,
        beta: float = 1.0,
        n_measurement_vars: int = None,
    ):
        super().__init__(
            name="USAD",
            n_features=n_features,
            window_size=window_size,
            n_measurement_vars=n_measurement_vars,
        )

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.hidden_dims = hidden_dims

        # Input dimension is flattened window
        input_dim = n_features * window_size

        # Shared encoder
        self.encoder = USADEncoder(input_dim, hidden_dims, latent_dim)

        # Two decoders
        self.decoder1 = USADDecoder(latent_dim, hidden_dims, input_dim)
        self.decoder2 = USADDecoder(latent_dim, hidden_dims, input_dim)

        # Training epoch counter
        self._epoch = 0
        self._n_epochs = 100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning AE1 reconstruction.

        Args:
            x: Input tensor [batch, n_features, window_size]

        Returns:
            Reconstruction [batch, n_features, window_size]
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        z = self.encoder(x_flat)
        recon1 = self.decoder1(z)
        return recon1.view(batch_size, self.n_features, self.window_size)

    def forward_ae1(self, x_flat: torch.Tensor) -> torch.Tensor:
        """AE1(W) = G1(E(W))."""
        z = self.encoder(x_flat)
        return self.decoder1(z)

    def forward_ae2(self, x_flat: torch.Tensor) -> torch.Tensor:
        """AE2(W) = G2(E(W))."""
        z = self.encoder(x_flat)
        return self.decoder2(z)

    def forward_ae2_ae1(self, x_flat: torch.Tensor) -> torch.Tensor:
        """AE2(AE1(W)) = G2(E(G1(E(W))))."""
        z = self.encoder(x_flat)
        recon1 = self.decoder1(z)
        z_recon = self.encoder(recon1)
        return self.decoder2(z_recon)

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Not used directly - training uses two-optimizer loop.

        This computes a combined loss for validation purposes only.
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        ae1_out = self.forward_ae1(x_flat)
        ae2_out = self.forward_ae2(x_flat)
        ae2_ae1_out = self.forward_ae2_ae1(x_flat)

        n = 1.0 / max(self._epoch, 1)

        loss_ae1 = n * F.mse_loss(ae1_out, x_flat) + (1 - n) * F.mse_loss(ae2_ae1_out, x_flat)
        loss_ae2 = n * F.mse_loss(ae2_out, x_flat) - (1 - n) * F.mse_loss(ae2_ae1_out, x_flat)

        total_loss = loss_ae1 + loss_ae2

        return total_loss, {
            'loss_ae1': loss_ae1,
            'loss_ae2': loss_ae2,
            'total_loss': total_loss,
        }

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute USAD anomaly scores.

        Score = α * ||x - AE1(x)|| + β * ||x - AE2(AE1(x))||
        Scored on measurement channels only.
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        ae1_out = self.forward_ae1(x_flat)
        ae2_ae1_out = self.forward_ae2_ae1(x_flat)

        # Reshape back
        ae1_recon = ae1_out.view(batch_size, self.n_features, self.window_size)
        ae2_ae1_recon = ae2_ae1_out.view(batch_size, self.n_features, self.window_size)

        # Score only on measurement channels
        n_m = self.n_measurement_vars
        error_ae1 = torch.mean((x[:, :n_m] - ae1_recon[:, :n_m]) ** 2, dim=(1, 2))
        error_ae2 = torch.mean((x[:, :n_m] - ae2_ae1_recon[:, :n_m]) ** 2, dim=(1, 2))

        anomaly_score = self.alpha * error_ae1 + self.beta * error_ae2
        return anomaly_score

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> "USAD":
        """Train USAD with two-optimizer adversarial training.

        Per the paper:
        - opt1 updates E + G1 with L_AE1
        - opt2 updates E + G2 with L_AE2
        - Epoch schedule: n = 1/epoch (harmonic)
        """
        self._n_epochs = epochs
        self._epoch = 0

        self.to(device)

        # Two separate optimizers (Bug fix #1)
        opt1 = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder1.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        opt2 = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder2.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self._epoch = epoch

            # Harmonic schedule: n = 1/epoch (Bug fix #4)
            n = 1.0 / epoch

            # Training
            train_loss = self._train_epoch_usad(
                train_loader, opt1, opt2, n, device
            )

            # Validation
            val_loss = self._evaluate_usad(val_loader, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and epoch % 10 == 0:
                print(
                    f"[{self.name}] Epoch {epoch:3d}/{epochs}: "
                    f"Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}"
                )

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        self._is_fitted = True
        self._train_stats['best_val_loss'] = best_val_loss
        self._train_stats['final_epoch'] = epoch

        return self

    def _train_epoch_usad(
        self,
        train_loader: DataLoader,
        opt1: torch.optim.Optimizer,
        opt2: torch.optim.Optimizer,
        n: float,
        device: torch.device,
    ) -> float:
        """Run one USAD training epoch with two-optimizer loop.

        Per the paper, each batch has two separate forward/backward passes:
        1. Forward pass 1: compute L_AE1, update E + G1
        2. Forward pass 2: compute L_AE2, update E + G2
        """
        self.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = self._extract_batch(batch, device)
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)

            # === Forward pass 1: Update E + G1 (Bug fix #5) ===
            opt1.zero_grad()

            ae1_out = self.forward_ae1(x_flat)
            ae2_ae1_out = self.forward_ae2_ae1(x_flat)

            # L_AE1 = (1/n)||W - AE1(W)|| + (1-1/n)||W - AE2(AE1(W))||
            loss_ae1 = (n * F.mse_loss(ae1_out, x_flat)
                        + (1 - n) * F.mse_loss(ae2_ae1_out, x_flat))

            loss_ae1.backward()
            opt1.step()

            # === Forward pass 2: Update E + G2 (Bug fix #5) ===
            opt2.zero_grad()

            # Fresh forward pass (Bug fix #5: separate forward passes)
            ae2_out = self.forward_ae2(x_flat)  # Bug fix #2: AE2(W) term
            ae2_ae1_out = self.forward_ae2_ae1(x_flat)

            # L_AE2 = (1/n)||W - AE2(W)|| - (1-1/n)||W - AE2(AE1(W))||
            # Bug fix #3: minus sign
            loss_ae2 = (n * F.mse_loss(ae2_out, x_flat)
                        - (1 - n) * F.mse_loss(ae2_ae1_out, x_flat))

            loss_ae2.backward()
            opt2.step()

            total_loss += (loss_ae1.item() + loss_ae2.item())
            n_batches += 1

        return total_loss / n_batches

    def _evaluate_usad(self, data_loader: DataLoader, device: torch.device) -> float:
        """Evaluate USAD on validation set."""
        self.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                x = self._extract_batch(batch, device)
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)

                ae1_out = self.forward_ae1(x_flat)
                ae2_ae1_out = self.forward_ae2_ae1(x_flat)

                # Validation loss: reconstruction quality
                loss = F.mse_loss(ae1_out, x_flat) + F.mse_loss(ae2_ae1_out, x_flat)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            'latent_dim': self.latent_dim,
            'alpha': self.alpha,
            'beta': self.beta,
        })
        return info
