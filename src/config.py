"""
Configuration management for CONGA experiments.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "ml-1m"
    data_dir: str = "data"
    max_seq_len: int = 50
    min_seq_len: int = 5
    train_batch_size: int = 256
    eval_batch_size: int = 256
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "sasrec"
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout_rate: float = 0.2
    
    # Graph-specific
    use_graph: bool = False
    graph_layers: int = 2
    graph_type: str = "gcn"  # gcn, gat, sage

    # CONGA nested-graph specific
    num_local_layers: int = 2
    num_global_layers: int = 1
    memory_bank_size: int = 1000
    
    # Contrastive learning
    use_contrastive: bool = False
    contrastive_weight: float = 0.1
    graph_cl_weight: float = 0.1
    temperature: float = 0.07
    augmentation_types: List[str] = field(default_factory=lambda: ["crop", "mask", "reorder"])
    
    # Continual learning
    use_continual: bool = False
    memory_size: int = 1000
    # Replay buffer size (alias used by CONGA/CONGAv2 configs)
    replay_buffer_size: int = 10000
    replay_ratio: float = 0.3
    distillation_weight: float = 0.5

    # CONGAv2 continual learning extensions
    use_ewc: bool = False
    ewc_lambda: float = 1000.0
    use_nested_learning: bool = False
    num_experts: int = 4
    importance_alpha: float = 0.6


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    optimizer: str = "adam"
    scheduler: str = "none"  # none, cosine, step
    warmup_steps: int = 0
    gradient_clip: float = 1.0
    
    # Loss function
    loss_type: str = "bce"  # bce, bpr, duo, infonce
    loss_config: dict = field(default_factory=dict)
    num_negatives: int = 1
    
    # Evaluation
    eval_interval: int = 10
    early_stopping_patience: int = 20
    metrics: List[str] = field(default_factory=lambda: ["hr@10", "ndcg@10", "mrr"])
    
    # Checkpointing
    save_dir: str = "experiments/checkpoints"
    log_dir: str = "experiments/logs"


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True
    experiment_name: str = "default"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "cuda"),
            use_amp=config_dict.get("use_amp", True),
            experiment_name=config_dict.get("experiment_name", "default"),
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "seed": self.seed,
            "device": self.device,
            "use_amp": self.use_amp,
            "experiment_name": self.experiment_name,
        }
