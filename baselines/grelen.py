"""GRELEN: Graph Relational Learning for Anomaly Detection.

Based on: Zhang et al. 2022 - "GRELEN: Multivariate Time Series Anomaly
Detection from the Perspective of Graph Relational Learning" (IJCAI 2022)

Architecture:
1. GumbelSoftmaxGraphLearner: Multi-head attention -> Gumbel-Softmax -> binary adjacency
2. DiffusionConv: sum_{k=0}^K theta_k * (D^-1 A)^k * X (bidirectional random walk)
3. DCGRUCell: GRU with diffusion convolutions replacing linear transforms
4. GRELEN: Full model with learned graph, DCGRU encoder, linear decoder

Scoring: (1-topo_weight) * recon_error + topo_weight * topo_score
where topo_score is node degree deviation from stored normal adjacency.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaselineModel


class NodeAttentionGraphLearner(nn.Module):
    """Learn binary adjacency via per-node attention + Gumbel-Softmax.

    Each node gets a query and key embedding. Attention scores between
    all pairs are converted to binary edges via Gumbel-Softmax.
    """

    def __init__(self, n_nodes: int, window_size: int, n_heads: int = 4,
                 head_dim: int = 32, temperature: float = 0.5):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.temperature = temperature
        self.scale = head_dim ** -0.5

        # Project each node's time-series to query/key
        self.W_q = nn.Linear(window_size, n_heads * head_dim)
        self.W_k = nn.Linear(window_size, n_heads * head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute adjacency matrix.

        Args:
            x: [B, N, W]

        Returns:
            A: [B, N, N] binary adjacency (soft during training)
        """
        B, N, W = x.shape

        Q = self.W_q(x).view(B, N, self.n_heads, self.head_dim)  # [B, N, H, D]
        K = self.W_k(x).view(B, N, self.n_heads, self.head_dim)

        # Attention: [B, H, N, N]
        Q = Q.permute(0, 2, 1, 3)  # [B, H, N, D]
        K = K.permute(0, 2, 1, 3)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Average over heads
        attn = attn.mean(dim=1)  # [B, N, N]

        # Gumbel-Softmax to get binary-like adjacency
        # Stack [attn, -attn] as logits for binary choice
        logits = torch.stack([attn, -attn], dim=-1)  # [B, N, N, 2]

        if self.training:
            A = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            A = A[..., 0]  # Take the "edge exists" probability
        else:
            A = (attn > 0).float()

        # Remove self-loops
        mask = 1.0 - torch.eye(N, device=x.device).unsqueeze(0)
        A = A * mask

        return A


class DiffusionConv(nn.Module):
    """Diffusion convolution: sum_{k=0}^K theta_k * T^k * X.

    Bidirectional: uses both D_o^-1 A (forward) and D_i^-1 A^T (backward).
    """

    def __init__(self, in_dim: int, out_dim: int, max_diffusion_step: int = 2):
        super().__init__()
        self.max_diffusion_step = max_diffusion_step
        # (2K+1) supports: K forward + K backward + identity
        n_supports = 2 * max_diffusion_step + 1
        self.weight = nn.Linear(n_supports * in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Diffusion convolution.

        Args:
            x: Node features [B, N, D]
            A: Adjacency matrix [B, N, N]

        Returns:
            Output [B, N, out_dim]
        """
        B, N, D = x.shape

        # Compute forward and backward transition matrices
        # Forward: D_out^-1 A
        d_out = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N, 1]
        T_f = A / d_out  # row-normalized

        # Backward: D_in^-1 A^T
        A_t = A.transpose(-2, -1)
        d_in = A_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        T_b = A_t / d_in

        # Collect diffusion supports
        supports = [x]  # identity (k=0)

        # Forward diffusion
        x_f = x
        for _ in range(self.max_diffusion_step):
            x_f = torch.bmm(T_f, x_f)  # [B, N, D]
            supports.append(x_f)

        # Backward diffusion
        x_b = x
        for _ in range(self.max_diffusion_step):
            x_b = torch.bmm(T_b, x_b)
            supports.append(x_b)

        # Concatenate all supports
        x_cat = torch.cat(supports, dim=-1)  # [B, N, (2K+1)*D]

        return self.weight(x_cat)  # [B, N, out_dim]


class DCGRUCell(nn.Module):
    """GRU cell with diffusion convolutions replacing linear transforms.

    GRU equations with DiffConv:
        r = sigma(DiffConv([x, h], A))
        u = sigma(DiffConv([x, h], A))
        c = tanh(DiffConv([x, r*h], A))
        h' = u * h + (1-u) * c
    """

    def __init__(self, input_dim: int, hidden_dim: int, max_diffusion_step: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Gates: reset (r) and update (u)
        self.gate_conv = DiffusionConv(
            input_dim + hidden_dim, hidden_dim * 2, max_diffusion_step
        )
        # Candidate
        self.cand_conv = DiffusionConv(
            input_dim + hidden_dim, hidden_dim, max_diffusion_step
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                A: torch.Tensor) -> torch.Tensor:
        """One step of DCGRU.

        Args:
            x: Input [B, N, input_dim]
            h: Hidden state [B, N, hidden_dim]
            A: Adjacency [B, N, N]

        Returns:
            New hidden state [B, N, hidden_dim]
        """
        # Gates
        xh = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gate_conv(xh, A))
        r, u = gates.chunk(2, dim=-1)

        # Candidate
        xrh = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.cand_conv(xrh, A))

        h_new = u * h + (1 - u) * c
        return h_new


class GRELEN(BaselineModel):
    """GRELEN: Graph Relational Learning for Anomaly Detection.

    Learns a graph structure via attention + Gumbel-Softmax, then uses
    DCGRU (diffusion convolutional GRU) to process the time series with
    the learned graph. Anomaly scoring combines reconstruction error and
    topology deviation from normal graph.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        n_hid: int = 64,
        n_heads: int = 4,
        head_dim: int = 32,
        max_diffusion_step: int = 2,
        temperature: float = 0.5,
        n_rnn_layers: int = 1,
        topo_weight: float = 0.5,
        n_measurement_vars: int = None,
    ):
        super().__init__(
            name="GRELEN",
            n_features=n_features,
            window_size=window_size,
            n_measurement_vars=n_measurement_vars,
        )

        self.n_hid = n_hid
        self.n_rnn_layers = n_rnn_layers
        self.topo_weight = topo_weight

        # Graph learner
        self.graph_learner = NodeAttentionGraphLearner(
            n_nodes=n_features,
            window_size=window_size,
            n_heads=n_heads,
            head_dim=head_dim,
            temperature=temperature,
        )

        # DCGRU layers
        self.dcgru_cells = nn.ModuleList()
        for i in range(n_rnn_layers):
            in_dim = 1 if i == 0 else n_hid  # each node has 1 feature per timestep
            self.dcgru_cells.append(
                DCGRUCell(in_dim, n_hid, max_diffusion_step)
            )

        # Output decoder: per-timestep prediction
        self.decoder = nn.Linear(n_hid, 1)

        # Normal adjacency buffer (set after training)
        self.register_buffer('A_normal', torch.zeros(n_features, n_features))
        self._A_accum = []

    def _encode_sequence(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Process sequence through DCGRU layers.

        Args:
            x: [B, N, W]
            A: [B, N, N]

        Returns:
            Hidden states [B, W, N, n_hid]
        """
        B, N, W = x.shape

        # Process each timestep
        all_hidden = []
        h = [torch.zeros(B, N, self.n_hid, device=x.device) for _ in range(self.n_rnn_layers)]

        for t in range(W):
            inp = x[:, :, t:t+1]  # [B, N, 1]
            for layer_idx, cell in enumerate(self.dcgru_cells):
                h[layer_idx] = cell(inp if layer_idx == 0 else h[layer_idx - 1], h[layer_idx], A)
            all_hidden.append(h[-1])  # [B, N, n_hid]

        return torch.stack(all_hidden, dim=1)  # [B, W, N, n_hid]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input [B, n_features, window_size]

        Returns:
            Reconstruction [B, n_features, window_size]
        """
        B = x.shape[0]

        # Learn graph
        A = self.graph_learner(x)  # [B, N, N]
        self._last_A = A

        # Encode with DCGRU
        hidden = self._encode_sequence(x, A)  # [B, W, N, n_hid]

        # Decode each timestep
        recon = self.decoder(hidden).squeeze(-1)  # [B, W, N]
        recon = recon.permute(0, 2, 1)  # [B, N, W]

        return recon

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """MSE reconstruction loss (measurement channels only)."""
        n_m = self.n_measurement_vars
        loss = F.mse_loss(recon[:, :n_m], x[:, :n_m])
        return loss, {'recon_loss': loss}

    def _compute_batch_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Combined recon error + topology deviation scoring.

        Score = (1 - topo_weight) * recon_error + topo_weight * topo_score
        """
        recon = self.forward(x)

        # Reconstruction error (measurement channels only)
        n_m = self.n_measurement_vars
        recon_error = torch.mean((x[:, :n_m] - recon[:, :n_m]) ** 2, dim=(1, 2))

        # Topology score: degree deviation from normal adjacency
        A = self._last_A  # [B, N, N]
        if self.A_normal.sum() > 0:
            # Node degree deviation
            deg_current = A.sum(dim=-1)  # [B, N]
            deg_normal = self.A_normal.sum(dim=-1).unsqueeze(0)  # [1, N]
            # Score on measurement nodes only
            topo_score = torch.mean(
                (deg_current[:, :n_m] - deg_normal[:, :n_m]) ** 2, dim=-1
            )
        else:
            topo_score = torch.zeros_like(recon_error)

        score = (1 - self.topo_weight) * recon_error + self.topo_weight * topo_score
        return score

    def _train_epoch(self, train_loader, optimizer, device):
        """Override to accumulate adjacency matrices during training."""
        self.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            x = self._extract_batch(batch, device)
            recon = self.forward(x)
            loss, _ = self.compute_loss(x, recon)
            loss.backward()
            optimizer.step()

            # Accumulate adjacency for normal reference
            self._A_accum.append(self._last_A.detach().mean(dim=0).cpu())

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches, {}

    def fit(self, train_loader, val_loader, epochs, device,
            learning_rate=1e-3, weight_decay=1e-5,
            early_stopping_patience=10, es_warmup=0, verbose=True):
        """Train and compute normal adjacency after training."""
        self._A_accum = []

        # Standard training
        result = super().fit(
            train_loader, val_loader, epochs, device,
            learning_rate, weight_decay, early_stopping_patience,
            es_warmup, verbose
        )

        # Compute mean normal adjacency from training
        if self._A_accum:
            A_mean = torch.stack(self._A_accum).mean(dim=0)
            self.A_normal.copy_(A_mean)
            if verbose:
                print(f"  Normal adjacency: mean degree = "
                      f"{self.A_normal.sum(dim=-1).mean():.2f}")
        self._A_accum = []

        return result

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update({
            'n_hid': self.n_hid,
            'n_rnn_layers': self.n_rnn_layers,
            'topo_weight': self.topo_weight,
        })
        return info
