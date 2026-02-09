"""GDN: Graph Deviation Network for Multivariate Time Series Anomaly Detection.

Based on: Deng & Hooi 2021 - "Graph Neural Network-Based Anomaly Detection in
Multivariate Time Series" (AAAI 2021)

Reference: https://github.com/d-ailin/GDN

Architecture:
- Learn sensor dependency graph via node embedding similarity
- Graph attention for feature aggregation
- FORECASTING-based: input [t-W..t-1] -> predict value at t (Bug fix #1)
- Per-node error normalized by validation median/IQR (Bug fix #3)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from .base import BaselineModel


class GraphStructureLearning(nn.Module):
    """Learn graph structure from node embeddings."""

    def __init__(self, n_nodes: int, embed_dim: int, topk: int = 10):
        super().__init__()
        self.n_nodes = n_nodes
        self.embed_dim = embed_dim
        self.topk = min(topk, n_nodes - 1)

        self.node_embedding = nn.Parameter(
            torch.randn(n_nodes, embed_dim) * 0.1
        )

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_emb = F.normalize(self.node_embedding, p=2, dim=-1)
        similarity = torch.mm(norm_emb, norm_emb.t())

        similarity = similarity - torch.eye(self.n_nodes, device=similarity.device) * 1e9

        topk_values, topk_indices = torch.topk(similarity, self.topk, dim=-1)

        adj = torch.zeros_like(similarity)
        for i in range(self.n_nodes):
            adj[i, topk_indices[i]] = F.softmax(topk_values[i], dim=-1)

        return adj, similarity


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for feature aggregation."""

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads

        self.W = nn.Linear(in_features, out_features)
        self.attention = nn.Parameter(torch.randn(n_heads, 2 * self.head_dim) * 0.1)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply graph attention.

        Args:
            x: Node features [batch, n_nodes, in_features]
            adj: Adjacency matrix [n_nodes, n_nodes]

        Returns:
            Updated features [batch, n_nodes, out_features]
        """
        batch_size, n_nodes, _ = x.shape

        h = self.W(x)
        h = h.view(batch_size, n_nodes, self.n_heads, self.head_dim)

        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)
        h_concat = torch.cat([h_i, h_j], dim=-1)

        e = torch.sum(h_concat * self.attention, dim=-1)
        e = self.leaky_relu(e)

        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(mask, float('-inf'))

        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        h_prime = torch.einsum('bijk,bjkd->bikd', alpha, h)
        h_prime = h_prime.reshape(batch_size, n_nodes, -1)

        return h_prime


class GDN(BaselineModel):
    """Graph Deviation Network for anomaly detection.

    Fixed implementation:
    - Forecasting model: predict next timestep (Bug fix #1)
    - Raw windowed features as input (Bug fix #2)
    - Validation-based normalization for scoring (Bug fix #3)
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        n_heads: int = 4,
        topk: int = 10,
        dropout: float = 0.1,
        n_measurement_vars: int = None,
    ):
        super().__init__(
            name="GDN",
            n_features=n_features,
            window_size=window_size,
            n_measurement_vars=n_measurement_vars,
        )

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.topk = topk

        # Graph structure learning
        self.graph_learner = GraphStructureLearning(
            n_nodes=n_features,
            embed_dim=embed_dim,
            topk=topk,
        )

        # Feature projection: raw window -> hidden_dim per node
        # Bug fix #2: use linear projection on raw features, not CNN
        # Input is W-1 timesteps (we hold out last for target)
        self.feature_proj = nn.Linear(window_size - 1, hidden_dim)

        # Graph attention
        self.gat = GraphAttentionLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Prediction head: forecast 1 value per node (Bug fix #1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # For storing learned graph
        self._adj = None

        # Validation-based normalization statistics (Bug fix #3)
        self.register_buffer('_val_median', torch.zeros(n_features))
        self.register_buffer('_val_iqr', torch.ones(n_features))
        self._val_stats_computed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for forecasting.

        Input: [batch, n_features, window_size]
        Uses x[:, :, :-1] as input, predicts x[:, :, -1]

        Returns:
            Prediction [batch, n_features, 1]
        """
        # Split input/target: use first W-1 steps as input
        x_input = x[:, :, :-1]  # [batch, n_features, W-1]

        # Learn graph structure
        adj, _ = self.graph_learner()
        self._adj = adj

        # Project each node's time series to hidden dim
        node_features = self.feature_proj(x_input)  # [batch, n_features, hidden_dim]

        # Graph attention aggregation
        graph_features = self.gat(node_features, adj)

        # Predict next timestep for each node
        predictions = self.predictor(graph_features)  # [batch, n_features, 1]

        return predictions

    def compute_loss(
        self,
        x: torch.Tensor,
        pred: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute forecasting loss.

        Args:
            x: Full input [batch, n_features, window_size] (target = last step)
            pred: Predicted output [batch, n_features, 1]
        """
        # Target is last timestep
        target = x[:, :, -1:]  # [batch, n_features, 1]

        mse_loss = F.mse_loss(pred, target, reduction='mean')

        # Graph regularization
        if self._adj is not None:
            graph_reg = torch.norm(self._adj, p=1) / (self.n_features ** 2)
        else:
            graph_reg = torch.tensor(0.0, device=x.device)

        total_loss = mse_loss + 0.01 * graph_reg

        return total_loss, {
            'mse_loss': mse_loss,
            'graph_reg': graph_reg,
            'total_loss': total_loss,
        }

    def _compute_val_normalization(self, val_loader, device):
        """Compute per-node median and IQR from validation set (Bug fix #3)."""
        self.eval()
        all_errors = []

        with torch.no_grad():
            for batch in val_loader:
                x = self._extract_batch(batch, device)
                pred = self.forward(x)
                target = x[:, :, -1:]

                # Per-node absolute error [batch, n_features]
                errors = torch.abs(pred - target).squeeze(-1)
                all_errors.append(errors.cpu())

        all_errors = torch.cat(all_errors, dim=0)  # [N, n_features]

        # Per-node median and IQR
        self._val_median = all_errors.median(dim=0).values.to(device)
        q75 = torch.quantile(all_errors.float(), 0.75, dim=0).to(device)
        q25 = torch.quantile(all_errors.float(), 0.25, dim=0).to(device)
        self._val_iqr = (q75 - q25).clamp(min=1e-8)
        self._val_stats_computed = True

    def compute_per_feature_residuals(self, data_loader, device):
        """Override: GDN forecasts next timestep, so residual = |target - pred|."""
        self.eval()
        all_residuals = []

        with torch.no_grad():
            for batch in data_loader:
                x = self._extract_batch(batch, device)
                pred = self.forward(x)  # [batch, n_features, 1]
                target = x[:, :, -1:]   # [batch, n_features, 1]

                n_m = self.n_measurement_vars
                # Per-feature absolute error: [batch, n_meas]
                residuals = torch.abs(
                    pred[:, :n_m] - target[:, :n_m]
                ).squeeze(-1)
                all_residuals.append(residuals.cpu().numpy())

        return np.concatenate(all_residuals, axis=0)

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores with per-node normalization (Bug fix #3).

        Per the paper: normalize each node's error by validation statistics,
        then aggregate via max across nodes.
        Only scores measurement channels.
        """
        pred = self.forward(x)
        target = x[:, :, -1:]  # [batch, n_features, 1]

        # Per-node absolute error
        errors = torch.abs(pred - target).squeeze(-1)  # [batch, n_features]

        n_m = self.n_measurement_vars

        if self._val_stats_computed:
            # Normalize by validation median/IQR (Bug fix #3)
            normalized = (errors[:, :n_m] - self._val_median[:n_m]) / self._val_iqr[:n_m]
        else:
            normalized = errors[:, :n_m]

        # Aggregate via max across measurement nodes (per paper)
        anomaly_score = normalized.max(dim=-1).values

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
    ) -> "GDN":
        """Train GDN and compute validation normalization statistics."""
        # Call parent fit for training
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
            train_loss, _ = self._train_epoch(train_loader, optimizer, device)
            val_loss, _ = self._evaluate(val_loader, device)

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

        # Compute validation normalization statistics (Bug fix #3)
        self._compute_val_normalization(val_loader, device)

        self._is_fitted = True
        self._train_stats['best_val_loss'] = best_val_loss
        self._train_stats['final_epoch'] = epoch

        return self

    def get_learned_graph(self) -> Optional[np.ndarray]:
        if self._adj is not None:
            return self._adj.detach().cpu().numpy()
        return None

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'topk': self.topk,
        })
        return info
