"""Baseline anomaly detection methods for DySTGAT comparison.

This package provides implementations of several baseline methods from
the multivariate time series anomaly detection literature:

- LSTM-VAE: LSTM Variational Autoencoder (Park et al. 2018)
- USAD: UnSupervised Anomaly Detection with dual autoencoders (Audibert et al. 2020)
- OmniAnomaly: Stochastic RNN with normalizing flows (Su et al. 2019)
- GDN: Graph Deviation Network (Deng & Hooi 2021)
- MTAD-GAT: Multi-scale Temporal Anomaly Detection with GAT (Zhao et al. 2020)
- DyEdgeGAT: Temporal-only DySTGAT (ablation baseline)

Usage:
    from baselines import create_baseline, list_baselines

    # List available methods
    print(list_baselines())

    # Create a model
    model = create_baseline(
        "lstm_vae",
        n_features=4,
        window_size=1024
    )

    # Train
    model.fit(train_loader, val_loader, epochs=100, device=device)

    # Compute anomaly scores
    scores = model.compute_anomaly_scores(test_loader, device)
"""

from typing import Dict, List, Optional, Type, Any

from .base import BaselineModel
from .lstm_vae import LSTMVAE
from .usad import USAD
from .omnianomaly import OmniAnomaly
from .gdn import GDN
from .mtad_gat import MTADGAT
from .dyedgegat import DyEdgeGAT

# Registry of baseline methods
_BASELINE_REGISTRY: Dict[str, Type[BaselineModel]] = {
    "lstm_vae": LSTMVAE,
    "usad": USAD,
    "omnianomaly": OmniAnomaly,
    "gdn": GDN,
    "mtad_gat": MTADGAT,
    "dyedgegat": DyEdgeGAT,
}

# Method descriptions
_BASELINE_DESCRIPTIONS: Dict[str, str] = {
    "lstm_vae": "LSTM-VAE (Park et al. 2018) - LSTM encoder-decoder with VAE regularization",
    "usad": "USAD (Audibert et al. 2020) - Dual autoencoder with adversarial training",
    "omnianomaly": "OmniAnomaly (Su et al. 2019) - Stochastic RNN with normalizing flows",
    "gdn": "GDN (Deng & Hooi 2021) - Graph Deviation Network with attention",
    "mtad_gat": "MTAD-GAT (Zhao et al. 2020) - Multi-scale Temporal + GAT",
    "dyedgegat": "DyEdgeGAT - DySTGAT without spectral view (temporal-only ablation)",
}

# Default hyperparameters for each method
_DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "lstm_vae": {
        "hidden_dim": 64,
        "latent_dim": 32,
        "num_layers": 2,
        "dropout": 0.1,
        "beta": 1.0,
    },
    "usad": {
        "hidden_dims": [128, 64],
        "latent_dim": 32,
        "alpha": 1.0,
        "beta": 1.0,
    },
    "omnianomaly": {
        "hidden_dim": 64,
        "latent_dim": 16,
        "n_flows": 4,
        "beta": 1.0,
    },
    "gdn": {
        "embed_dim": 64,
        "hidden_dim": 64,
        "n_heads": 4,
        "topk": 10,
        "dropout": 0.1,
    },
    "mtad_gat": {
        "hidden_dim": 64,
        "n_heads": 4,
        "n_temporal_layers": 3,
        "dropout": 0.1,
        "forecast_horizon": 1,
    },
    "dyedgegat": {
        "node_encoder_hidden": 64,
        "gnn_embed_dim": 40,
        "num_gnn_layers": 2,
        "gnn_type": "gin",
        "temp_edge_hid_dim": 100,
        "temp_node_embed_dim": 16,
        "topk": 20,
        "dropout": 0.3,
    },
}


def list_baselines() -> List[str]:
    """List all available baseline methods.

    Returns:
        List of baseline method names
    """
    return list(_BASELINE_REGISTRY.keys())


def get_baseline_description(method: str) -> str:
    """Get description of a baseline method.

    Args:
        method: Baseline method name

    Returns:
        Description string
    """
    if method not in _BASELINE_DESCRIPTIONS:
        raise ValueError(f"Unknown baseline: {method}. Available: {list_baselines()}")
    return _BASELINE_DESCRIPTIONS[method]


def get_default_hyperparams(method: str) -> Dict[str, Any]:
    """Get default hyperparameters for a baseline method.

    Args:
        method: Baseline method name

    Returns:
        Dict of default hyperparameters
    """
    if method not in _DEFAULT_HYPERPARAMS:
        raise ValueError(f"Unknown baseline: {method}. Available: {list_baselines()}")
    return _DEFAULT_HYPERPARAMS[method].copy()


def create_baseline(
    method: str,
    n_features: int,
    window_size: int,
    **kwargs
) -> BaselineModel:
    """Create a baseline model instance.

    Args:
        method: Baseline method name (e.g., "lstm_vae", "usad", "gdn")
        n_features: Number of input features/channels
        window_size: Temporal window size
        **kwargs: Additional method-specific hyperparameters

    Returns:
        Initialized baseline model

    Example:
        >>> model = create_baseline("lstm_vae", n_features=4, window_size=1024)
        >>> model = create_baseline("gdn", n_features=4, window_size=1024, topk=5)
    """
    if method not in _BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline method: {method}. "
            f"Available methods: {list_baselines()}"
        )

    # Get default hyperparameters and override with kwargs
    hyperparams = get_default_hyperparams(method)
    hyperparams.update(kwargs)

    # Create model
    model_class = _BASELINE_REGISTRY[method]
    model = model_class(
        n_features=n_features,
        window_size=window_size,
        **hyperparams
    )

    return model


def register_baseline(
    name: str,
    model_class: Type[BaselineModel],
    description: str = "",
    default_hyperparams: Optional[Dict[str, Any]] = None,
) -> None:
    """Register a new baseline method.

    Args:
        name: Method name for registry
        model_class: Model class (must inherit from BaselineModel)
        description: Method description
        default_hyperparams: Default hyperparameters
    """
    if not issubclass(model_class, BaselineModel):
        raise TypeError(f"Model class must inherit from BaselineModel")

    _BASELINE_REGISTRY[name] = model_class
    _BASELINE_DESCRIPTIONS[name] = description
    _DEFAULT_HYPERPARAMS[name] = default_hyperparams or {}


# Export public API
__all__ = [
    # Base class
    "BaselineModel",
    # Model classes
    "LSTMVAE",
    "USAD",
    "OmniAnomaly",
    "GDN",
    "MTADGAT",
    "DyEdgeGAT",
    # Factory functions
    "create_baseline",
    "list_baselines",
    "get_baseline_description",
    "get_default_hyperparams",
    "register_baseline",
]
