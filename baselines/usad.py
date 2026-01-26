"""USAD: UnSupervised Anomaly Detection on Multivariate Time Series.

Based on: Audibert et al. 2020 - "USAD: UnSupervised Anomaly Detection on
Multivariate Time Series"

Architecture:
- Shared encoder E
- Two decoders G1, G2
- Phase 1: Train AE1 = G1(E(x)), AE2 = G2(E(G1(E(x))))
- Phase 2: Adversarial training
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

    This model uses adversarial training between two autoencoders:
    - AE1: Standard autoencoder (E + G1)
    - AE2: Tries to distinguish real from reconstructed (E + G2)

    The adversarial objective makes AE1 produce better reconstructions
    while AE2 learns to detect anomalies.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_dims: list = None,
        latent_dim: int = 32,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        """Initialize USAD model.

        Args:
            n_features: Number of input features
            window_size: Temporal window size
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
            alpha: Weight for AE1 reconstruction loss in anomaly score
            beta: Weight for AE2 discriminator loss in anomaly score
        """
        super().__init__(
            name="USAD",
            n_features=n_features,
            window_size=window_size,
        )

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta

        # Input dimension is flattened window
        input_dim = n_features * window_size

        # Shared encoder
        self.encoder = USADEncoder(input_dim, hidden_dims, latent_dim)

        # Two decoders
        self.decoder1 = USADDecoder(latent_dim, hidden_dims, input_dim)
        self.decoder2 = USADDecoder(latent_dim, hidden_dims, input_dim)

        # Training epoch counter for adversarial schedule
        self._epoch = 0
        self._n_epochs = 100  # Will be set during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning AE1 reconstruction.

        Args:
            x: Input tensor [batch, n_features, window_size]

        Returns:
            Reconstruction [batch, n_features, window_size]
        """
        batch_size = x.shape[0]

        # Flatten input
        x_flat = x.view(batch_size, -1)

        # AE1: E -> G1
        z = self.encoder(x_flat)
        recon1 = self.decoder1(z)

        # Reshape back
        recon = recon1.view(batch_size, self.n_features, self.window_size)

        return recon

    def forward_full(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass returning both AE outputs.

        Args:
            x: Input tensor [batch, n_features, window_size]

        Returns:
            Tuple of (x_flat, recon1, recon2)
        """
        batch_size = x.shape[0]

        # Flatten
        x_flat = x.view(batch_size, -1)

        # AE1: x -> E -> G1
        z = self.encoder(x_flat)
        recon1 = self.decoder1(z)

        # AE2: recon1 -> E -> G2
        z_recon = self.encoder(recon1)
        recon2 = self.decoder2(z_recon)

        return x_flat, recon1, recon2

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute USAD training loss.

        The loss varies based on training phase (epoch number).
        Phase 1: Standard AE reconstruction
        Phase 2: Adversarial training

        Args:
            x: Original input [batch, n_features, window_size]
            recon: Not used (we compute full forward here)

        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Full forward pass
        x_flat, recon1, recon2 = self.forward_full(x)

        # Reconstruction losses
        loss_ae1 = F.mse_loss(recon1, x_flat)  # AE1 wants to reconstruct x
        loss_ae2_real = F.mse_loss(recon2, x_flat)  # AE2 also reconstructs from AE1's output

        # Adversarial component: schedule based on epoch
        # n = epoch / total_epochs
        n = self._epoch / max(self._n_epochs, 1)

        # AE1 loss: minimize reconstruction + fool AE2
        # AE2 loss: minimize its reconstruction + catch AE1's fakes
        loss_ae1_total = (1 - n) * loss_ae1 + n * loss_ae2_real
        loss_ae2_total = (1 - n) * loss_ae2_real + n * loss_ae1

        # Combined loss (we train both simultaneously)
        total_loss = loss_ae1_total + loss_ae2_total

        loss_components = {
            'loss_ae1': loss_ae1,
            'loss_ae2': loss_ae2_real,
            'total_loss': total_loss,
        }

        return total_loss, loss_components

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute USAD anomaly scores.

        Score = α * ||x - AE1(x)|| + β * ||x - AE2(AE1(x))||

        Args:
            x: Input tensor [batch, n_features, window_size]

        Returns:
            Anomaly scores [batch]
        """
        x_flat, recon1, recon2 = self.forward_full(x)

        # Per-sample reconstruction errors
        error_ae1 = torch.mean((x_flat - recon1) ** 2, dim=-1)
        error_ae2 = torch.mean((x_flat - recon2) ** 2, dim=-1)

        # Combined anomaly score
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
        """Train USAD with epoch-based adversarial scheduling.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            device: Training device
            learning_rate: Learning rate
            weight_decay: L2 regularization
            early_stopping_patience: Patience for early stopping
            verbose: Print progress

        Returns:
            Self
        """
        self._n_epochs = epochs
        self._epoch = 0

        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self._epoch = epoch

            # Training
            train_loss = self._train_epoch_usad(train_loader, optimizer, device)

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
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> float:
        """Run one USAD training epoch."""
        self.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Handle PyG Data objects
            if hasattr(batch, 'x'):
                x = batch.x.to(device)
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
                n_nodes = x.shape[0] // batch_size
                x = x.view(batch_size, n_nodes, -1)
            else:
                x = batch[0].to(device)

            # Compute loss
            loss, _ = self.compute_loss(x, None)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _evaluate_usad(self, data_loader: DataLoader, device: torch.device) -> float:
        """Evaluate USAD on validation set."""
        self.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                if hasattr(batch, 'x'):
                    x = batch.x.to(device)
                    batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
                    n_nodes = x.shape[0] // batch_size
                    x = x.view(batch_size, n_nodes, -1)
                else:
                    x = batch[0].to(device)

                loss, _ = self.compute_loss(x, None)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def get_model_info(self) -> Dict:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            'latent_dim': self.latent_dim,
            'alpha': self.alpha,
            'beta': self.beta,
        })
        return info
