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
from torch_geometric.data import Data

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
        n_measurement_vars: int = None,
    ):
        """Initialize DyEdgeGAT model.

        Args:
            n_features: Number of measurement features (nodes)
            window_size: Temporal window size
            node_encoder_hidden: Node encoder hidden dimension
            gnn_embed_dim: GNN embedding dimension
            num_gnn_layers: Number of GNN layers
            gnn_type: GNN type (gin, gat, gcn)
            temp_edge_hid_dim: Temporal edge inference hidden dim
            temp_node_embed_dim: Temporal node embedding dim
            topk: Number of neighbors in inferred graph
            dropout: Dropout rate
            ocvar_dim: Control variable dimension (auto-detected if n_measurement_vars set)
            n_measurement_vars: Number of measurement variables
        """
        # DyEdgeGAT uses only measurement vars as nodes, control vars separately
        if n_measurement_vars is not None and n_measurement_vars < n_features:
            ocvar_dim = n_features - n_measurement_vars
            actual_n_features = n_measurement_vars
        else:
            actual_n_features = n_features

        super().__init__(
            name="DyEdgeGAT",
            n_features=actual_n_features,
            window_size=window_size,
            n_measurement_vars=actual_n_features,
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

    def forward(self, x: torch.Tensor, control: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through DySTGAT (temporal only).

        Args:
            x: Input [batch, n_features, window_size] (measurement vars only)
            control: Control variables [batch, ocvar_dim, window_size] or None

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

        # Build control tensor for DySTGAT: [batch*ocvar_dim, window_size]
        if self.ocvar_dim > 0 and control is not None:
            # control: [batch, ocvar_dim, window_size]
            ctrl_flat = control.reshape(batch_size * self.ocvar_dim, self.window_size)
        else:
            ctrl_flat = torch.zeros(batch_size, 0, device=x.device)

        # Construct PyG Data object expected by DySTGAT.forward()
        data = Data(
            x=x_flat,
            c=ctrl_flat,
            edge_index=torch.empty(2, 0, dtype=torch.long, device=x.device),
            batch=batch_tensor,
        )

        # Forward through DySTGAT
        recon_flat = self._model(
            data,
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

    def compute_per_feature_residuals(self, data_loader, device):
        """Override: DyEdgeGAT needs separate measurement/control extraction."""
        self.eval()
        all_residuals = []

        with torch.no_grad():
            for batch in data_loader:
                x = self._extract_measurement(batch, device)
                control = self._extract_control(batch, device)

                recon = self.forward(x, control=control)

                # Per-feature MAE averaged over time: [batch, n_features]
                residuals = torch.mean(torch.abs(x - recon), dim=2)
                all_residuals.append(residuals.cpu().numpy())

        import numpy as np
        return np.concatenate(all_residuals, axis=0)

    def compute_anomaly_scores(self, data_loader, device, return_labels=False):
        """Override to handle DyEdgeGAT's separate measurement/control extraction."""
        self.eval()
        scores_list = []
        labels_list = []

        with torch.no_grad():
            for batch in data_loader:
                x = self._extract_measurement(batch, device)
                control = self._extract_control(batch, device)

                if hasattr(batch, 'y') and batch.y is not None:
                    labels_list.append(batch.y.cpu().numpy())

                batch_scores = self._compute_batch_anomaly_scores_with_ctrl(x, control)
                scores_list.append(batch_scores.cpu().numpy())

        import numpy as np
        scores = np.concatenate(scores_list, axis=0)
        if return_labels and labels_list:
            labels = np.concatenate(labels_list, axis=0)
            return scores, labels
        return scores

    def _compute_batch_anomaly_scores_with_ctrl(
        self, x: torch.Tensor, control: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute anomaly scores using DySTGAT's topology-aware scoring.

        Args:
            x: Measurement input [batch, n_features, window_size]
            control: Control variables [batch, ocvar_dim, window_size] or None

        Returns:
            Anomaly scores [batch]
        """
        self._init_dystgat(x.device)

        batch_size = x.shape[0]

        # Reshape for DySTGAT
        x_flat = x.reshape(batch_size * self.n_features, self.window_size)
        batch_tensor = torch.arange(batch_size, device=x.device).repeat_interleave(
            self.n_features
        )

        # Build control tensor for DySTGAT: [batch*ocvar_dim, window_size]
        if self.ocvar_dim > 0 and control is not None:
            ctrl_flat = control.reshape(batch_size * self.ocvar_dim, self.window_size)
        else:
            ctrl_flat = torch.zeros(batch_size, 0, device=x.device)

        # Construct PyG Data object expected by DySTGAT.forward()
        data = Data(
            x=x_flat,
            c=ctrl_flat,
            edge_index=torch.empty(2, 0, dtype=torch.long, device=x.device),
            batch=batch_tensor,
        )

        # Forward with graph
        outputs = self._model(
            data,
            return_graph=True,
        )

        if isinstance(outputs, tuple):
            recon_flat, edge_index, edge_attr, _ = outputs
        else:
            recon_flat = outputs

        target_flat = x_flat

        if hasattr(self._model, 'compute_anomaly_scores_per_sample'):
            scores = self._model.compute_anomaly_scores_per_sample(
                target_flat, recon_flat, edge_index, edge_attr
            )
        else:
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

    def _extract_control(self, batch, device):
        """Extract control variables from batch."""
        if hasattr(batch, 'c') and batch.c is not None:
            c = batch.c.to(device)
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            if c.dim() == 2:
                n_ctrl = c.shape[0] // batch_size
                c = c.view(batch_size, n_ctrl, -1)
            return c
        return None

    def _extract_measurement(self, batch, device):
        """Extract measurement variables from batch."""
        if hasattr(batch, 'x'):
            x = batch.x.to(device)
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            n_nodes = x.shape[0] // batch_size
            return x.view(batch_size, n_nodes, -1)
        return batch[0].to(device)

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

            x = self._extract_measurement(batch, device)
            control = self._extract_control(batch, device)

            recon = self.forward(x, control=control)
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
                x = self._extract_measurement(batch, device)
                control = self._extract_control(batch, device)

                recon = self.forward(x, control=control)
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
        self._init_dystgat(device)
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
        if self._model is None:
            self._init_dystgat(torch.device('cpu'))
        return self._model.parameters()

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
