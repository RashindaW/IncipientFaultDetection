"""DyEdgeGAT: Temporal-only DySTGAT baseline (without spectral view).

This is a wrapper around DySTGAT that disables the spectral view,
serving as an ablation study to isolate the contribution of the
spectral branch in DySTGAT.

The DyEdgeGAT baseline uses:
- Dynamic temporal edge inference
- GNN message passing on temporal graph
- Reconstruction-based anomaly detection

Without the spectral view, this model does not benefit from:
- Frequency-domain representation
- Multi-view graph divergence
- Spectral-temporal fusion
"""

from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

from .base import BaselineModel


class DyEdgeGAT(BaselineModel):
    """DyEdgeGAT: Temporal-only variant of DySTGAT.

    This is a convenience wrapper that initializes DySTGAT
    with use_spectral_view=False.

    Note: This requires the full DySTGAT codebase to be available.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        # DySTGAT temporal parameters
        node_encoder_hidden: int = 64,
        gnn_embed_dim: int = 40,
        num_gnn_layers: int = 2,
        gnn_type: str = "gin",
        temp_edge_hid_dim: int = 100,
        temp_node_embed_dim: int = 16,
        topk: int = 20,
        dropout: float = 0.3,
        # Additional control
        ocvar_dim: int = 0,
    ):
        """Initialize DyEdgeGAT model.

        Args:
            n_features: Number of input features (nodes)
            window_size: Temporal window size
            node_encoder_hidden: Node encoder hidden dimension
            gnn_embed_dim: GNN embedding dimension
            num_gnn_layers: Number of GNN layers
            gnn_type: GNN type (gin, gat, gcn)
            temp_edge_hid_dim: Temporal edge inference hidden dim
            temp_node_embed_dim: Temporal node embedding dim
            topk: Number of neighbors in inferred graph
            dropout: Dropout rate
            ocvar_dim: Control variable dimension
        """
        super().__init__(
            name="DyEdgeGAT",
            n_features=n_features,
            window_size=window_size,
        )

        self.node_encoder_hidden = node_encoder_hidden
        self.gnn_embed_dim = gnn_embed_dim
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type
        self.temp_edge_hid_dim = temp_edge_hid_dim
        self.temp_node_embed_dim = temp_node_embed_dim
        self.topk = topk
        self.dropout = dropout
        self.ocvar_dim = ocvar_dim

        # Lazy initialization of DySTGAT
        self._model = None
        self._cfg_initialized = False

    def _init_dystgat(self, device: torch.device):
        """Lazily initialize the DySTGAT model."""
        if self._model is not None:
            return

        try:
            from dystgat.src.config import cfg
            from dystgat.src.model.dystgat import DySTGAT as DySTGATModel

            # Set config
            cfg.set_dataset_params(
                n_nodes=self.n_features,
                window_size=self.window_size,
                ocvar_dim=self.ocvar_dim,
                pred_horizon=0,
                task="reconstruction"
            )
            cfg.device = str(device)
            cfg.validate()

            # Initialize DySTGAT without spectral view
            self._model = DySTGATModel(
                # Input/output dimensions
                feat_input_node=1,
                feat_target_node=1,
                feat_input_edge=1,
                # Node encoder
                node_encoder_type="gru",
                node_encoder_mode="univariate",
                contr_encoder_type="gru",
                # Temporal graph inference
                infer_temporal_edge=True,
                temp_edge_hid_dim=self.temp_edge_hid_dim,
                temp_edge_embed_dim=1,
                temporal_window=5,
                temporal_kernel=5,
                # Node embeddings
                temp_node_embed_dim=self.temp_node_embed_dim,
                # Static graph
                infer_static_graph=True,
                feat_edge_hid_dim=128,
                topk=self.topk,
                learn_sys=True,
                # GNN
                num_gnn_layers=self.num_gnn_layers,
                gnn_embed_dim=self.gnn_embed_dim,
                gnn_type=self.gnn_type,
                dropout=self.dropout,
                # Normalization
                do_encoder_norm=True,
                do_gnn_norm=True,
                do_decoder_norm=True,
                encoder_norm_type="layer",
                gnn_norm_type="layer",
                decoder_norm_type="layer",
                # Decoder
                recon_hidden_dim=16,
                num_recon_layers=1,
                edge_aggr="temp",
                act="relu",
                aug_control=True,
                flip_output=True,
                # SPECTRAL VIEW DISABLED
                use_spectral_view=False,
                # Task
                task="reconstruction",
                pred_horizon=0,
            )

            self._model.to(device)
            self._cfg_initialized = True

        except ImportError as e:
            raise ImportError(
                "DyEdgeGAT requires the DySTGAT model to be available. "
                f"Import error: {e}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DySTGAT (temporal only).

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Reconstruction [batch, n_features, window_size]
        """
        self._init_dystgat(x.device)

        batch_size = x.shape[0]

        # Reshape to DySTGAT format: [batch*n_features, window_size]
        x_flat = x.view(batch_size * self.n_features, self.window_size)

        # Create batch tensor for PyG
        batch_tensor = torch.arange(batch_size, device=x.device).repeat_interleave(
            self.n_features
        )

        # Create minimal control variables if needed
        if self.ocvar_dim > 0:
            control = torch.zeros(batch_size, self.ocvar_dim, device=x.device)
        else:
            control = None

        # Forward through DySTGAT
        recon_flat = self._model(
            x_flat,
            batch=batch_tensor,
            control=control,
            return_graph=False,
        )

        # Reshape back to [batch, n_features, window_size]
        recon = recon_flat.view(batch_size, self.n_features, self.window_size)

        return recon

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> tuple:
        """Compute reconstruction loss.

        Args:
            x: Original input
            recon: Reconstruction

        Returns:
            Tuple of (loss, loss_components)
        """
        mse_loss = torch.nn.functional.mse_loss(recon, x, reduction='mean')

        return mse_loss, {
            'mse_loss': mse_loss,
            'total_loss': mse_loss,
        }

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores using DySTGAT's topology-aware scoring.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Anomaly scores [batch]
        """
        self._init_dystgat(x.device)

        batch_size = x.shape[0]

        # Reshape for DySTGAT
        x_flat = x.view(batch_size * self.n_features, self.window_size)
        batch_tensor = torch.arange(batch_size, device=x.device).repeat_interleave(
            self.n_features
        )

        control = None
        if self.ocvar_dim > 0:
            control = torch.zeros(batch_size, self.ocvar_dim, device=x.device)

        # Forward with graph
        outputs = self._model(
            x_flat,
            batch=batch_tensor,
            control=control,
            return_graph=True,
        )

        if isinstance(outputs, tuple):
            recon_flat, edge_index, edge_attr, _ = outputs
        else:
            recon_flat = outputs

        # Compute per-sample anomaly scores
        target_flat = x_flat

        if hasattr(self._model, 'compute_anomaly_scores_per_sample'):
            scores = self._model.compute_anomaly_scores_per_sample(
                target_flat, recon_flat, edge_index, edge_attr
            )
        else:
            # Fallback to MSE
            recon = recon_flat.view(batch_size, self.n_features, self.window_size)
            scores = torch.mean((x - recon) ** 2, dim=(1, 2))

        return scores

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> "DyEdgeGAT":
        """Train the DyEdgeGAT model.

        Note: For proper training, it's recommended to use the
        train_dystgat.py script with --use-spectral-view flag omitted.

        Args:
            train_loader: Training data
            val_loader: Validation data
            epochs: Number of epochs
            device: Training device
            learning_rate: Learning rate
            weight_decay: L2 regularization
            early_stopping_patience: Early stopping patience
            verbose: Print progress

        Returns:
            Self
        """
        self._init_dystgat(device)
        self.to(device)

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Training
            train_loss = self._train_epoch_dystgat(train_loader, optimizer, device)

            # Validation
            val_loss = self._evaluate_dystgat(val_loader, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
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
            self._model.load_state_dict(best_state)

        self._is_fitted = True
        self._train_stats['best_val_loss'] = best_val_loss
        self._train_stats['final_epoch'] = epoch

        return self

    def _train_epoch_dystgat(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> float:
        """Training epoch for DyEdgeGAT."""
        self._model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if hasattr(batch, 'x'):
                x = batch.x.to(device)
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
                n_nodes = x.shape[0] // batch_size
                x = x.view(batch_size, n_nodes, -1)
            else:
                x = batch[0].to(device)

            recon = self.forward(x)
            loss, _ = self.compute_loss(x, recon)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _evaluate_dystgat(self, data_loader: DataLoader, device: torch.device) -> float:
        """Evaluation for DyEdgeGAT."""
        self._model.eval()
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

                recon = self.forward(x)
                loss, _ = self.compute_loss(x, recon)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        if self._model is None:
            raise RuntimeError("Model not initialized. Call forward() first.")

        checkpoint = {
            'name': self.name,
            'n_features': self.n_features,
            'window_size': self.window_size,
            'state_dict': self._model.state_dict(),
            'is_fitted': self._is_fitted,
            'train_stats': self._train_stats,
            'config': {
                'node_encoder_hidden': self.node_encoder_hidden,
                'gnn_embed_dim': self.gnn_embed_dim,
                'num_gnn_layers': self.num_gnn_layers,
                'gnn_type': self.gnn_type,
                'temp_edge_hid_dim': self.temp_edge_hid_dim,
                'temp_node_embed_dim': self.temp_node_embed_dim,
                'topk': self.topk,
                'dropout': self.dropout,
                'ocvar_dim': self.ocvar_dim,
            }
        }
        torch.save(checkpoint, path)

    def load(self, path: str, device: Optional[torch.device] = None) -> "DyEdgeGAT":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)

        # Initialize model if needed
        if device is not None:
            self._init_dystgat(device)

        self._model.load_state_dict(checkpoint['state_dict'])
        self._is_fitted = checkpoint.get('is_fitted', True)
        self._train_stats = checkpoint.get('train_stats', {})

        return self

    def to(self, device: torch.device) -> "DyEdgeGAT":
        """Move model to device."""
        if self._model is not None:
            self._model.to(device)
        return self

    def train(self, mode: bool = True):
        """Set training mode."""
        if self._model is not None:
            self._model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        if self._model is not None:
            self._model.eval()
        return self

    def parameters(self):
        """Get model parameters."""
        if self._model is not None:
            return self._model.parameters()
        return iter([])

    def get_model_info(self) -> Dict:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            'gnn_embed_dim': self.gnn_embed_dim,
            'num_gnn_layers': self.num_gnn_layers,
            'gnn_type': self.gnn_type,
            'topk': self.topk,
            'spectral_view': False,  # Key difference from DySTGAT
        })
        return info
