"""Abstract base class for baseline anomaly detection models.

All baseline methods inherit from BaselineModel and implement a common interface
for training, inference, and anomaly scoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaselineModel(ABC, nn.Module):
    """Abstract base class for baseline anomaly detection models.

    All baseline implementations must:
    1. Inherit from this class
    2. Implement the abstract methods
    3. Follow the same training/evaluation interface

    Attributes:
        name: Human-readable model name for logging
        n_features: Number of input features/channels
        window_size: Temporal window size
    """

    def __init__(self, name: str, n_features: int, window_size: int,
                 n_measurement_vars: int = None):
        """Initialize the baseline model.

        Args:
            name: Model name for logging and identification
            n_features: Number of input features (measurement + control)
            window_size: Temporal window size (e.g., 1024 for IMS-raw)
            n_measurement_vars: Number of measurement variables for scoring.
                If None, all features are used for scoring.
        """
        super().__init__()
        self.name = name
        self.n_features = n_features
        self.window_size = window_size
        self.n_measurement_vars = n_measurement_vars or n_features

        # Track training state
        self._is_fitted = False
        self._train_stats: Dict[str, Any] = {}

    def _extract_batch(self, batch, device: torch.device) -> torch.Tensor:
        """Extract input tensor from a batch, concatenating x + c if available.

        Args:
            batch: PyG Data batch or tuple batch
            device: Device to move tensors to

        Returns:
            Input tensor of shape [batch_size, n_features, window_size]
        """
        if hasattr(batch, 'x'):
            x = batch.x.to(device)
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            n_nodes = x.shape[0] // batch_size
            x = x.view(batch_size, n_nodes, -1)  # [B, N_meas, W]

            # Concatenate control variables if available
            if hasattr(batch, 'c') and batch.c is not None:
                c = batch.c.to(device)  # [B, C, W] or [B*C, W]
                if c.dim() == 2:
                    # Flat format: [B*C, W] -> [B, C, W]
                    n_ctrl = c.shape[0] // batch_size
                    c = c.view(batch_size, n_ctrl, -1)
                x = torch.cat([x, c], dim=1)  # [B, N_meas+C, W]
        else:
            x = batch[0].to(device)

        return x

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, n_features, window_size]
               or [batch_size * n_features, window_size] (flattened)

        Returns:
            Reconstruction or prediction tensor of same shape as input
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the training loss.

        Args:
            x: Original input tensor
            recon: Reconstructed output tensor
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple of (total_loss, loss_components_dict)
            loss_components_dict contains individual loss terms for logging
        """
        pass

    def compute_anomaly_scores(
        self,
        data_loader: DataLoader,
        device: torch.device,
        return_labels: bool = False
    ) -> np.ndarray:
        """Compute anomaly scores for all samples in the data loader.

        Args:
            data_loader: DataLoader containing samples to score
            device: Device to run inference on
            return_labels: If True, also return labels

        Returns:
            Anomaly scores array of shape [n_samples]
            If return_labels=True, returns (scores, labels) tuple
        """
        self.eval()
        scores_list = []
        labels_list = []

        with torch.no_grad():
            for batch in data_loader:
                x = self._extract_batch(batch, device)

                if hasattr(batch, 'y') and batch.y is not None:
                    labels_list.append(batch.y.cpu().numpy())
                elif not hasattr(batch, 'x') and len(batch) > 1:
                    labels_list.append(batch[1].cpu().numpy())

                # Compute per-sample scores
                batch_scores = self._compute_batch_anomaly_scores(x)
                scores_list.append(batch_scores.cpu().numpy())

        scores = np.concatenate(scores_list, axis=0)

        if return_labels and labels_list:
            labels = np.concatenate(labels_list, axis=0)
            return scores, labels
        return scores

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores for a batch of samples.

        Default implementation uses reconstruction error on measurement
        channels only (excludes control variables from scoring).
        Override this method for model-specific scoring.

        Args:
            x: Input tensor of shape [batch_size, n_features, window_size]

        Returns:
            Anomaly scores tensor of shape [batch_size]
        """
        recon = self.forward(x)
        # Score only on measurement channels
        n_m = self.n_measurement_vars
        mse = torch.mean((x[:, :n_m] - recon[:, :n_m]) ** 2, dim=(1, 2))
        return mse

    def compute_per_feature_residuals(
        self,
        data_loader: DataLoader,
        device: torch.device,
    ) -> np.ndarray:
        """Compute per-feature mean absolute residuals for each window.

        Used for unified IQR-normalized scoring (GDN's scoring function applied
        to all baselines, per DyEdgeGAT paper Section V.B).

        Args:
            data_loader: DataLoader containing samples
            device: Device to run inference on

        Returns:
            Per-feature residuals array of shape [N, n_measurement_vars]
        """
        self.eval()
        all_residuals = []

        with torch.no_grad():
            for batch in data_loader:
                x = self._extract_batch(batch, device)
                recon = self.forward(x)

                n_m = self.n_measurement_vars
                # Per-feature MAE averaged over time: [batch, n_meas]
                residuals = torch.mean(
                    torch.abs(x[:, :n_m] - recon[:, :n_m]), dim=2
                )
                all_residuals.append(residuals.cpu().numpy())

        return np.concatenate(all_residuals, axis=0)

    @staticmethod
    def iqr_normalize_scores(
        residuals: np.ndarray,
        val_median: np.ndarray,
        val_iqr: np.ndarray,
    ) -> np.ndarray:
        """Normalize per-feature residuals using validation IQR and aggregate.

        Per-feature normalization followed by mean aggregation across features
        (paper Section V.A).  Mean reflects the system-wide anomaly level.

        Args:
            residuals: Per-feature residuals [N, n_meas]
            val_median: Validation median per feature [n_meas]
            val_iqr: Validation IQR per feature [n_meas]

        Returns:
            Per-window scalar scores [N]
        """
        normalized = (residuals - val_median) / val_iqr
        scores = np.mean(normalized, axis=1)
        return scores

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 10,
        es_warmup: int = 0,
        verbose: bool = True
    ) -> "BaselineModel":
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            device: Device to train on
            learning_rate: Optimizer learning rate
            weight_decay: L2 regularization weight
            early_stopping_patience: Epochs to wait before early stopping
            es_warmup: Epoch before which early stopping is disabled
            verbose: Whether to print training progress

        Returns:
            Self for method chaining
        """
        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=10
        )

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_metrics = self._train_epoch(
                train_loader, optimizer, device
            )

            # Validation
            val_loss, val_metrics = self._evaluate(val_loader, device)

            scheduler.step(val_loss)

            # Early stopping
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

            if epoch >= es_warmup and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_state is not None:
            self.load_state_dict(best_state)

        self._is_fitted = True
        self._train_stats['best_val_loss'] = best_val_loss
        self._train_stats['final_epoch'] = epoch

        return self

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            device: Device to train on

        Returns:
            Tuple of (epoch_loss, metrics_dict)
        """
        self.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            x = self._extract_batch(batch, device)

            # Forward pass
            recon = self.forward(x)

            # Compute loss
            loss, _ = self.compute_loss(x, recon)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches, {}

    def _evaluate(
        self,
        data_loader: DataLoader,
        device: torch.device
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on a dataset.

        Args:
            data_loader: Data loader to evaluate on
            device: Device to run on

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                x = self._extract_batch(batch, device)

                recon = self.forward(x)
                loss, _ = self.compute_loss(x, recon)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches, {}

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'name': self.name,
            'n_features': self.n_features,
            'window_size': self.window_size,
            'state_dict': self.state_dict(),
            'is_fitted': self._is_fitted,
            'train_stats': self._train_stats,
        }
        torch.save(checkpoint, path)

    def load(self, path: str, device: Optional[torch.device] = None) -> "BaselineModel":
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model to

        Returns:
            Self for method chaining
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])
        self._is_fitted = checkpoint.get('is_fitted', True)
        self._train_stats = checkpoint.get('train_stats', {})
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            'name': self.name,
            'n_features': self.n_features,
            'window_size': self.window_size,
            'n_parameters': sum(p.numel() for p in self.parameters()),
            'is_fitted': self._is_fitted,
        }
