"""
Real-time prediction service that loads models and generates predictions.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict
import lightgbm as lgb

from src.config import Config
from src.model import ModelRegistry
from src.features import FeatureService
from src.schemas import Prediction
from src.utils import get_logger


class Predictor:
    """
    Loads active model and generates predictions at 1-second cadence.
    Maintains prediction history for monitoring and retraining triggers.
    """
    
    def __init__(
        self, 
        config: Config,
        model_registry: ModelRegistry,
        feature_service: FeatureService
    ):
        self.config = config
        self.model_registry = model_registry
        self.feature_service = feature_service
        self.logger = get_logger("predictor")
        
        # Model state
        self.model: Optional[lgb.Booster] = None
        self.model_version: Optional[str] = None
        self.model_metadata: Optional[Dict] = None
        
        # Prediction history
        self.prediction_history = []
        
        # Load active model
        self.load_active_model()
    
    def load_active_model(self) -> bool:
        """Load the active model from registry."""
        self.model = self.model_registry.load_model("active")
        self.model_metadata = self.model_registry.load_metadata("active")
        
        if self.model is None:
            self.logger.warning("no_active_model_found")
            return False
        
        self.model_version = self.model_metadata.get('model_version')
        
        self.logger.info(
            "model_loaded",
            version=self.model_version,
            metrics=self.model_metadata.get('metrics')
        )
        
        return True
    
    def reload_model_if_updated(self) -> bool:
        """Check if model has been updated and reload if necessary."""
        current_active_version = self.model_registry.get_active_version()
        
        if current_active_version != self.model_version:
            self.logger.info(
                "model_update_detected",
                old_version=self.model_version,
                new_version=current_active_version
            )
            return self.load_active_model()
        
        return False
    
    def predict(
        self, 
        df_1m: pd.DataFrame,
        df_1s_recent: pd.DataFrame = None
    ) -> Optional[Prediction]:
        """
        Generate prediction for current timestamp.
        
        Args:
            df_1m: Historical 1-minute bars
            df_1s_recent: Recent 1-second bars (optional)
            
        Returns:
            Prediction object or None
        """
        if self.model is None:
            self.logger.warning("prediction_skipped_no_model")
            return None
        
        # Compute features
        X_live = self.feature_service.compute_live_features(df_1m, df_1s_recent)
        
        if X_live.empty:
            self.logger.warning("prediction_skipped_no_features")
            return None
        
        try:
            # Generate prediction
            pred_value = self.model.predict(X_live)[0]
            
            prediction = Prediction(
                timestamp=datetime.now(),
                symbol=self.config.data.symbol,
                predicted_return=float(pred_value),
                model_version=self.model_version,
                features=X_live.iloc[0].to_dict()
            )
            
            # Store in history
            self.prediction_history.append(prediction)
            
            # Keep limited history (last 1000 predictions)
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            self.logger.debug(
                "prediction_generated",
                predicted_return=pred_value,
                model_version=self.model_version
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error("prediction_error", error=str(e))
            return None
    
    def get_prediction_statistics(self) -> Dict:
        """Get statistics on recent predictions."""
        if not self.prediction_history:
            return {}
        
        predictions = [p.predicted_return for p in self.prediction_history]
        
        stats = {
            'count': len(predictions),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'positive_pct': float(sum(p > 0 for p in predictions) / len(predictions) * 100)
        }
        
        return stats
    
    def clear_history(self):
        """Clear prediction history."""
        self.prediction_history = []
        self.logger.info("prediction_history_cleared")
