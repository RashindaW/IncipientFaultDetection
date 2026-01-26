"""GDN: Graph Deviation Network for Multivariate Time Series Anomaly Detection.

Based on: Deng & Hooi 2021 - "Graph Neural Network-Based Anomaly Detection in
Multivariate Time Series"

Architecture:
- Learn sensor dependency graph via node embedding similarity
- Graph attention for feature aggregation
- Forecasting-based anomaly detection
- Attention-weighted prediction errors for scoring
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

    def __init__(
        self,
        n_nodes: int,
        embed_dim: int,
        topk: int = 10,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.embed_dim = embed_dim
        self.topk = min(topk, n_nodes - 1)

        # Learnable node embeddings
        self.node_embedding = nn.Parameter(
            torch.randn(n_nodes, embed_dim) * 0.1
        )

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute adjacency matrix from embeddings.

        Returns:
            Tuple of (adjacency_matrix, attention_weights)
        """
        # Cosine similarity
        norm_emb = F.normalize(self.node_embedding, p=2, dim=-1)
        similarity = torch.mm(norm_emb, norm_emb.t())

        # Remove self-loops
        similarity = similarity - torch.eye(self.n_nodes, device=similarity.device) * 1e9

        # Top-k selection
        topk_values, topk_indices = torch.topk(similarity, self.topk, dim=-1)

        # Create sparse adjacency
        adj = torch.zeros_like(similarity)
        for i in range(self.n_nodes):
            adj[i, topk_indices[i]] = F.softmax(topk_values[i], dim=-1)

        return adj, similarity


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for feature aggregation."""

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

        self.W = nn.Linear(in_features, out_features)
        self.attention = nn.Parameter(torch.randn(n_heads, 2 * self.head_dim) * 0.1)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Apply graph attention.

        Args:
            x: Node features [batch, n_nodes, in_features]
            adj: Adjacency matrix [n_nodes, n_nodes]

        Returns:
            Updated features [batch, n_nodes, out_features]
        """
        batch_size, n_nodes, _ = x.shape

        # Linear transformation
        h = self.W(x)  # [batch, n_nodes, out_features]
        h = h.view(batch_size, n_nodes, self.n_heads, self.head_dim)

        # Compute attention scores
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)
        h_concat = torch.cat([h_i, h_j], dim=-1)  # [batch, n, n, heads, 2*head_dim]

        # Attention coefficients
        e = torch.sum(h_concat * self.attention, dim=-1)  # [batch, n, n, heads]
        e = self.leaky_relu(e)

        # Mask with adjacency
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(mask, float('-inf'))

        # Softmax attention
        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        # Aggregate
        h_prime = torch.einsum('bijk,bjkd->bikd', alpha, h)
        h_prime = h_prime.reshape(batch_size, n_nodes, -1)

        return h_prime


class FeatureEncoder(nn.Module):
    """Encode temporal features for each node."""

    def __init__(
        self,
        window_size: int,
        hidden_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3)
        )
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal features.

        Args:
            x: Input [batch, n_nodes, window_size]

        Returns:
            Node features [batch, n_nodes, hidden_dim]
        """
        batch_size, n_nodes, window_size = x.shape

        # Process each node
        x = x.view(batch_size * n_nodes, 1, window_size)

        for conv in self.conv_layers:
            x = F.relu(conv(x))

        # Pool to single vector
        x = self.pool(x).squeeze(-1)

        x = x.view(batch_size, n_nodes, -1)
        return x


class GDN(BaselineModel):
    """Graph Deviation Network for anomaly detection.

    This model:
    1. Learns sensor dependency graph via node embeddings
    2. Uses graph attention for feature aggregation
    3. Forecasts next values based on graph context
    4. Computes attention-weighted deviation scores
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
    ):
        """Initialize GDN model.

        Args:
            n_features: Number of sensors/nodes
            window_size: Temporal window size
            embed_dim: Node embedding dimension
            hidden_dim: Hidden layer dimension
            n_heads: Number of attention heads
            topk: Number of neighbors in learned graph
            dropout: Dropout rate
        """
        super().__init__(
            name="GDN",
            n_features=n_features,
            window_size=window_size,
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

        # Feature encoder
        self.feature_encoder = FeatureEncoder(
            window_size=window_size,
            hidden_dim=hidden_dim,
        )

        # Graph attention
        self.gat = GraphAttentionLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Prediction head (forecast next window)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, window_size),
        )

        # For storing learned graph
        self._adj = None
        self._attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for reconstruction/forecasting.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Prediction [batch, n_features, window_size]
        """
        # Learn graph structure
        adj, sim = self.graph_learner()
        self._adj = adj
        self._attention_weights = sim

        # Encode temporal features
        node_features = self.feature_encoder(x)  # [batch, n_nodes, hidden_dim]

        # Graph attention aggregation
        graph_features = self.gat(node_features, adj)

        # Predict output
        predictions = self.predictor(graph_features)  # [batch, n_nodes, window_size]

        return predictions

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute forecasting loss.

        Args:
            x: Original input (target for forecasting)
            recon: Predicted output

        Returns:
            Tuple of (loss, loss_components)
        """
        # MSE loss
        mse_loss = F.mse_loss(recon, x, reduction='mean')

        # Graph regularization (encourage sparsity)
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

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted deviation scores.

        Args:
            x: Input [batch, n_features, window_size]

        Returns:
            Anomaly scores [batch]
        """
        # Get predictions
        pred = self.forward(x)

        # Per-node deviation
        deviation = (x - pred) ** 2  # [batch, n_nodes, window_size]
        node_scores = deviation.mean(dim=-1)  # [batch, n_nodes]

        # Weight by graph importance (node degree/attention)
        if self._adj is not None:
            # Node importance = sum of outgoing attention
            node_importance = self._adj.sum(dim=1)  # [n_nodes]
            node_importance = F.softmax(node_importance, dim=0)

            # Weighted sum
            weighted_scores = node_scores * node_importance.unsqueeze(0)
            anomaly_score = weighted_scores.sum(dim=-1)
        else:
            anomaly_score = node_scores.mean(dim=-1)

        return anomaly_score

    def get_learned_graph(self) -> Optional[np.ndarray]:
        """Get the learned adjacency matrix."""
        if self._adj is not None:
            return self._adj.detach().cpu().numpy()
        return None

    def get_model_info(self) -> Dict:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'topk': self.topk,
        })
        return info
