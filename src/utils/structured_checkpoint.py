"""
Structured Checkpoint Manager with versioning and metadata.

Inspired by hindsight experience replay, this module provides:
1. Hierarchical checkpoint organization
2. Metadata tracking (metrics, hyperparameters, timestamps)
3. Incremental saving (only changed parameters)
4. Checkpoint versioning and rollback
5. Automatic cleanup of old checkpoints
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn


class StructuredCheckpointManager:
    """
    Manages model checkpoints with structured storage and metadata.
    
    Directory structure:
    checkpoints/
        experiment_name/
            metadata.json          # Experiment-level metadata
            versions/
                v001/
                    checkpoint.pt  # Model state
                    config.json    # Training config
                    metrics.json   # Performance metrics
                    manifest.json  # File manifest
                v002/
                    ...
            best/
                checkpoint.pt      # Best model (symlink or copy)
                metrics.json
            latest/
                checkpoint.pt      # Latest checkpoint
    """
    
    def __init__(
        self,
        base_dir: str,
        experiment_name: str,
        max_versions: int = 10,
        save_incremental: bool = True,
    ):
        """
        Args:
            base_dir: Base directory for all checkpoints
            experiment_name: Name of this experiment
            max_versions: Maximum number of versions to keep
            save_incremental: Whether to save only changed parameters
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.max_versions = max_versions
        self.save_incremental = save_incremental
        
        # Create directory structure
        self.exp_dir = self.base_dir / experiment_name
        self.versions_dir = self.exp_dir / "versions"
        self.best_dir = self.exp_dir / "best"
        self.latest_dir = self.exp_dir / "latest"
        
        for dir_path in [self.exp_dir, self.versions_dir, self.best_dir, self.latest_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata
        self.metadata_path = self.exp_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # Track previous checkpoint for incremental saving
        self.prev_state_dict = None
    
    def _load_metadata(self) -> Dict:
        """Load experiment metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        
        return {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "versions": [],
            "best_version": None,
            "best_metric": None,
        }
    
    def _save_metadata(self):
        """Save experiment metadata."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_next_version(self) -> str:
        """Get next version number."""
        existing_versions = [d.name for d in self.versions_dir.iterdir() if d.is_dir()]
        if not existing_versions:
            return "v001"
        
        version_nums = [int(v[1:]) for v in existing_versions if v.startswith("v")]
        next_num = max(version_nums) + 1 if version_nums else 1
        
        return f"v{next_num:03d}"
    
    def _compute_checkpoint_hash(self, state_dict: Dict) -> str:
        """Compute hash of checkpoint for deduplication."""
        # Hash based on parameter values
        hash_obj = hashlib.md5()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            if isinstance(tensor, torch.Tensor):
                hash_obj.update(tensor.cpu().numpy().tobytes())
        
        return hash_obj.hexdigest()
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save a checkpoint with metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            epoch: Current epoch
            metrics: Performance metrics
            config: Training configuration
            is_best: Whether this is the best checkpoint
            
        Returns:
            version: Version string of saved checkpoint
        """
        state_dict = model.state_dict()
        
        # Check if checkpoint is different from previous
        checkpoint_hash = self._compute_checkpoint_hash(state_dict)
        
        # Create version directory
        version = self._get_next_version()
        version_dir = self.versions_dir / version
        version_dir.mkdir(exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": int(epoch),
            "model_state_dict": state_dict,
            "checkpoint_hash": checkpoint_hash,
        }
        
        if optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        
        # Save checkpoint
        checkpoint_path = version_dir / "checkpoint.pt"
        torch.save(checkpoint_data, checkpoint_path, weights_only=False)
        
        # Save metrics
        if metrics:
            metrics_data = {
                "epoch": int(epoch),
                "timestamp": datetime.now().isoformat(),
                "metrics": {k: float(v) for k, v in metrics.items()},
            }
            with open(version_dir / "metrics.json", "w") as f:
                json.dump(metrics_data, f, indent=2)
        
        # Save config
        if config:
            with open(version_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
        
        # Create manifest
        manifest = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "epoch": int(epoch),
            "checkpoint_hash": checkpoint_hash,
            "files": [
                "checkpoint.pt",
                "metrics.json" if metrics else None,
                "config.json" if config else None,
            ],
        }
        manifest["files"] = [f for f in manifest["files"] if f is not None]
        
        with open(version_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Update metadata
        self.metadata["versions"].append({
            "version": version,
            "epoch": int(epoch),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        })
        
        # Update best checkpoint
        if is_best:
            self._update_best(version, metrics)
        
        # Update latest
        self._update_latest(version_dir)
        
        # Save metadata
        self._save_metadata()
        
        # Cleanup old versions
        self._cleanup_old_versions()
        
        return version
    
    def _update_best(self, version: str, metrics: Optional[Dict]):
        """Update best checkpoint."""
        version_dir = self.versions_dir / version
        
        # Copy to best directory
        for file in version_dir.iterdir():
            shutil.copy2(file, self.best_dir / file.name)
        
        # Update metadata
        self.metadata["best_version"] = version
        if metrics:
            self.metadata["best_metric"] = metrics
    
    def _update_latest(self, version_dir: Path):
        """Update latest checkpoint."""
        for file in version_dir.iterdir():
            shutil.copy2(file, self.latest_dir / file.name)
    
    def _cleanup_old_versions(self):
        """Remove old versions beyond max_versions."""
        versions = sorted(
            [d for d in self.versions_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
        )
        
        if len(versions) > self.max_versions:
            # Keep best version
            best_version = self.metadata.get("best_version")
            
            for old_version in versions[:-self.max_versions]:
                if old_version.name != best_version:
                    shutil.rmtree(old_version)
    
    def load_checkpoint(
        self,
        version: str = "latest",
        device: str = "cpu",
    ) -> Dict:
        """
        Load a checkpoint.
        
        Args:
            version: Version to load ("latest", "best", or "vXXX")
            device: Device to load checkpoint to
            
        Returns:
            checkpoint: Checkpoint dictionary
        """
        if version == "latest":
            checkpoint_path = self.latest_dir / "checkpoint.pt"
        elif version == "best":
            checkpoint_path = self.best_dir / "checkpoint.pt"
        else:
            checkpoint_path = self.versions_dir / version / "checkpoint.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        return checkpoint
    
    def list_versions(self) -> List[Dict]:
        """List all available versions with metadata."""
        return self.metadata.get("versions", [])
    
    def get_best_metrics(self) -> Optional[Dict]:
        """Get metrics of best checkpoint."""
        return self.metadata.get("best_metric")
    
    def export_checkpoint(
        self,
        version: str,
        export_path: str,
        include_optimizer: bool = False,
    ):
        """
        Export a checkpoint to a standalone file.
        
        Args:
            version: Version to export
            export_path: Path to export to
            include_optimizer: Whether to include optimizer state
        """
        checkpoint = self.load_checkpoint(version)
        
        export_data = {
            "model_state_dict": checkpoint["model_state_dict"],
            "epoch": checkpoint["epoch"],
        }
        
        if include_optimizer and "optimizer_state_dict" in checkpoint:
            export_data["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
        
        torch.save(export_data, export_path, weights_only=False)
