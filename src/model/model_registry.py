"""
Model registry for versioned model storage and promotion.
"""
import shutil
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import lightgbm as lgb

from src.config import Config
from src.utils import get_logger, ensure_dir


class ModelRegistry:
    """
    Manages versioned model storage with metadata tracking.
    Supports staging → active promotion with symlinks.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("model_registry")
        
        self.models_dir = Path(config.model_registry.models_dir)
        ensure_dir(self.models_dir)
        
        self.active_link = self.models_dir / "active"
        self.staging_link = self.models_dir / "staging"
    
    def register(
        self, 
        model: lgb.Booster, 
        metadata: Dict,
        status: str = "candidate"
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model: Trained LightGBM model
            metadata: Model metadata dict
            status: Model status ('candidate', 'staging', 'active')
            
        Returns:
            model_version: Version string
        """
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / model_version
        ensure_dir(model_dir)
        
        # Save model as pickle
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata['model_version'] = model_version
        metadata['status'] = status
        
        meta_path = model_dir / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Only log to file, not console
        self.logger.debug(
            "model_registered",
            version=model_version,
            status=status,
            metrics=metadata.get('metrics')
        )
        
        # Cleanup old models
        self._cleanup_old_models()
        
        return model_version
    
    def promote(self, model_version: str, target: str = "staging"):
        """
        Promote model to staging or active.
        
        Args:
            model_version: Model version to promote
            target: 'staging' or 'active'
        """
        model_dir = self.models_dir / model_version
        
        if not model_dir.exists():
            raise ValueError(f"Model version {model_version} not found")
        
        # Update metadata status
        meta_path = model_dir / "meta.json"
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['status'] = target
        metadata[f'{target}_promotion_time'] = datetime.now().isoformat()
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create/update symlink
        target_link = self.staging_link if target == "staging" else self.active_link
        
        # Remove existing link/directory
        if target_link.exists():
            if target_link.is_symlink():
                target_link.unlink()
            elif target_link.is_dir():
                shutil.rmtree(target_link)
            else:
                target_link.unlink()
        
        # Create symlink (Windows requires special handling)
        try:
            target_link.symlink_to(model_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Fallback: copy directory if symlink fails on Windows
            shutil.copytree(model_dir, target_link)
        
        self.logger.info(
            "model_promoted",
            version=model_version,
            target=target
        )
    
    def load_model(self, version: str = "active") -> Optional[lgb.Booster]:
        """
        Load model by version or link name.
        
        Args:
            version: Model version or 'active'/'staging'
            
        Returns:
            Loaded LightGBM model or None
        """
        if version == "active":
            model_dir = self.active_link
        elif version == "staging":
            model_dir = self.staging_link
        else:
            model_dir = self.models_dir / version
        
        if not model_dir.exists():
            return None
        
        model_path = model_dir / "model.pkl"
        
        if not model_path.exists():
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            self.logger.error("model_load_error", error=str(e))
            return None
    
    def load_metadata(self, version: str = "active") -> Optional[Dict]:
        """Load model metadata."""
        if version == "active":
            model_dir = self.active_link
        elif version == "staging":
            model_dir = self.staging_link
        else:
            model_dir = self.models_dir / version
        
        if not model_dir.exists():
            return None
        
        meta_path = model_dir / "meta.json"
        
        if not meta_path.exists():
            return None
        
        with open(meta_path, 'r') as f:
            return json.load(f)
    
    def list_models(self) -> List[Dict]:
        """List all registered models with metadata."""
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name in ['active', 'staging']:
                meta_path = model_dir / "meta.json"
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                        models.append(metadata)
        
        # Sort by training time
        models.sort(key=lambda x: x.get('training_time', ''), reverse=True)
        
        return models
    
    def get_active_version(self) -> Optional[str]:
        """Get active model version."""
        if not self.active_link.exists():
            return None
        
        metadata = self.load_metadata("active")
        return metadata.get('model_version') if metadata else None
    
    def _cleanup_old_models(self):
        """Remove old model versions beyond retention limit."""
        keep_n = self.config.model_registry.keep_last_n_models
        models = self.list_models()
        
        if len(models) <= keep_n:
            return
        
        # Keep most recent models
        models_to_delete = models[keep_n:]
        
        for metadata in models_to_delete:
            version = metadata['model_version']
            model_dir = self.models_dir / version
            
            # Don't delete if it's currently active or staging
            if metadata.get('status') in ['active', 'staging']:
                continue
            
            if model_dir.exists():
                shutil.rmtree(model_dir)
                self.logger.info("old_model_deleted", version=version)
