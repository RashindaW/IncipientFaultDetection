"""
Example usage of DyEdgeGAT with topology-aware anomaly scoring.

This script demonstrates how to:
1. Initialize the configuration
2. Create a DyEdgeGAT model
3. Perform forward pass
4. Compute topology-aware anomaly scores
"""

import torch
from dyedgegat.src.config import cfg
from dyedgegat.src.model.dyedgegat import DyEdgeGAT


def main():
    # ========== Step 1: Configure the model ==========
    # Set dataset parameters (example from Pronto dataset)
    cfg.set_dataset_params(
        n_nodes=17,        # Number of sensor nodes
        window_size=15,    # Sliding window size
        ocvar_dim=4        # Operating condition variables (control inputs)
    )
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Validate configuration
    cfg.validate()
    
    print(f"Configuration set: {cfg.dataset.n_nodes} nodes, "
          f"window size {cfg.dataset.window_size}, "
          f"device {cfg.device}")
    
    # ========== Step 2: Initialize the model ==========
    model = DyEdgeGAT(
        feat_input_node=1,           # Each sensor has 1 measurement variable
        feat_target_node=1,          # Reconstructing 1 variable per sensor
        feat_input_edge=1,           # Edge features are scalar attention values
        node_encoder_type='gru',
        node_encoder_mode='univariate',  # Separate GRU per sensor
        contr_encoder_type='gru',
        infer_temporal_edge=True,
        temp_edge_hid_dim=100,
        temp_edge_embed_dim=1,
        temporal_window=5,           # Temporal window for edge dynamics
        temporal_kernel=5,           # CNN kernel size for smoothing
        use_time_encoding=True,
        time_dim=5,
        temp_node_embed_dim=16,
        infer_static_graph=True,
        feat_edge_hid_dim=128,
        topk=20,                     # Keep top-20 edges per node
        learn_sys=True,
        num_gnn_layers=2,
        gnn_embed_dim=40,
        gnn_type='gin',              # Use GIN as in the paper
        dropout=0.3,
        do_encoder_norm=True,
        do_gnn_norm=True,
        do_decoder_norm=True,
        encoder_norm_type='layer',   # Changed to match paper
        gnn_norm_type='layer',       # Changed to match paper
        decoder_norm_type='layer',   # Changed to match paper
        recon_hidden_dim=16,
        num_recon_layers=1,
        edge_aggr='dot',
        act='relu',
        aug_control=True,
        flip_output=True,            # Default is now True (paper semantics)
    )
    
    model = model.to(cfg.device)
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # ========== Step 3: Create dummy batch data ==========
    batch_size = 4
    n_nodes = cfg.dataset.n_nodes
    window_size = cfg.dataset.window_size
    
    # Create a dummy batch (in practice, this comes from your data loader)
    class DummyBatch:
        def __init__(self):
            # Sensor measurements: [batch*n_nodes, window_size]
            self.x = torch.randn(batch_size * n_nodes, window_size, device=cfg.device)
            
            # Operating condition variables: [batch*n_nodes, ocvar_dim, window_size]
            self.c = torch.randn(batch_size, cfg.dataset.ocvar_dim, window_size, device=cfg.device)
            
            # Batch assignment: [batch*n_nodes]
            self.batch = torch.repeat_interleave(
                torch.arange(batch_size, device=cfg.device), 
                n_nodes
            )
            
            # Fully connected edge index (for feature graph initialization)
            # Each batch has a separate fully connected graph
            edge_indices = []
            for b in range(batch_size):
                src = []
                dst = []
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        src.append(b * n_nodes + i)
                        dst.append(b * n_nodes + j)
                edge_indices.append(torch.tensor([src, dst], device=cfg.device))
            self.edge_index = torch.cat(edge_indices, dim=1)
    
    batch = DummyBatch()
    print(f"\nBatch created: {batch_size} samples, "
          f"{batch.x.shape[0]} total nodes, "
          f"{batch.edge_index.shape[1]} total edges")
    
    # ========== Step 4: Forward pass ==========
    model.eval()
    with torch.no_grad():
        # Get reconstruction and learned graph
        recon, edge_index, edge_weight = model(batch, return_graph=True)
    
    print(f"\nForward pass completed:")
    print(f"  Input shape: {batch.x.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Learned graph edges: {edge_index.shape[1]}")
    print(f"  Edge weights shape: {edge_weight.shape}")
    
    # ========== Step 5: Compute topology-aware anomaly scores ==========
    
    # Method 1: Single scalar score for the entire batch
    anomaly_score = model.compute_topology_aware_anomaly_score(
        x_true=batch.x.unsqueeze(-1),  # [B*N, W, 1]
        x_recon=recon,
        edge_index=edge_index,
        edge_weight=edge_weight.view(-1)
    )
    print(f"\nTopology-aware anomaly score (entire batch): {anomaly_score.item():.6f}")
    
    # Method 2: Per-sample anomaly scores
    sample_scores = model.compute_anomaly_scores_per_sample(
        x_true=batch.x.unsqueeze(-1),
        x_recon=recon,
        edge_index=edge_index,
        edge_weight=edge_weight.view(-1)
    )
    print(f"\nPer-sample anomaly scores:")
    for i, score in enumerate(sample_scores):
        print(f"  Sample {i}: {score.item():.6f}")
    
    # ========== Step 6: Anomaly detection example ==========
    # In practice, you would determine this threshold from validation data
    threshold = sample_scores.mean() + 2 * sample_scores.std()
    
    anomalies = sample_scores > threshold
    print(f"\nAnomaly detection (threshold={threshold.item():.6f}):")
    for i, is_anomaly in enumerate(anomalies):
        status = "ANOMALY" if is_anomaly else "NORMAL"
        print(f"  Sample {i}: {status} (score: {sample_scores[i].item():.6f})")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

