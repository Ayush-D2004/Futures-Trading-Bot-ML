"""
LightGBM model training pipeline with validation and metrics.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from typing import Tuple, Dict
import json

from src.config import Config
from src.features import FeatureService
from src.utils import get_logger, hash_dataframe, calculate_sharpe_ratio


class Trainer:
    """
    Trains LightGBM regression model on 12-hour rolling window.
    Uses walk-forward validation with early stopping.
    """
    
    def __init__(self, config: Config, feature_service: FeatureService):
        self.config = config
        self.feature_service = feature_service
        self.logger = get_logger("trainer")
        
        self.lgb_params = self._get_lgb_params()
    
    def _get_lgb_params(self) -> dict:
        """Extract LightGBM parameters from config."""
        params = self.config.training.lightgbm_params.model_dump()
        
        # Remove non-LightGBM params
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        
        return params, early_stopping_rounds
    
    def train(self, df_1m: pd.DataFrame) -> Tuple[lgb.Booster, Dict]:
        """
        Train model on 1-minute bars.
        
        Args:
            df_1m: DataFrame with 1-minute OHLCV bars
            
        Returns:
            model: Trained LightGBM model
            metadata: Training metrics and metadata
        """
        self.logger.debug("training_started", num_rows=len(df_1m))
        
        # Compute features and labels
        X, y = self.feature_service.compute_train_features(df_1m)
        
        if len(X) < self.config.training.min_training_rows:
            raise ValueError(
                f"Insufficient training data: {len(X)} < {self.config.training.min_training_rows}"
            )
        
        # Walk-forward split: last 20% for validation
        val_split = self.config.training.validation_split
        split_idx = int(len(X) * (1 - val_split))
        
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.logger.debug(
            "data_split",
            train_size=len(X_train),
            val_size=len(X_val)
        )
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        lgb_params, early_stopping_rounds = self._get_lgb_params()
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50)
        ]
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        # Evaluate
        metrics = self._evaluate_model(model, X_train, y_train, X_val, y_val)
        
        # Compute dataset hash for reproducibility
        dataset_hash = hash_dataframe(df_1m)
        
        # Build metadata
        metadata = {
            'training_time': datetime.now().isoformat(),
            'dataset_hash': dataset_hash,
            'feature_spec': self.feature_service.feature_spec,
            'params': lgb_params,
            'metrics': metrics,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'num_features': len(X.columns),
            'feature_names': list(X.columns)
        }
        
        self.logger.debug("training_completed", metrics=metrics)
        
        return model, metadata
    
    def _evaluate_model(
        self, 
        model: lgb.Booster, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series
    ) -> Dict:
        """Evaluate model performance on train and validation sets."""
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Regression metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        # Directional accuracy
        train_dir_acc = self._directional_accuracy(y_train, y_train_pred)
        val_dir_acc = self._directional_accuracy(y_val, y_val_pred)
        
        # Simulated trading metrics
        val_sharpe = self._simulate_sharpe(y_val, y_val_pred)
        
        metrics = {
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse),
            'train_mae': float(train_mae),
            'val_mae': float(val_mae),
            'train_directional_accuracy': float(train_dir_acc),
            'val_directional_accuracy': float(val_dir_acc),
            'val_sharpe': float(val_sharpe)
        }
        
        return metrics
    
    def _directional_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate directional prediction accuracy."""
        correct = ((y_true > 0) & (y_pred > 0)) | ((y_true < 0) & (y_pred < 0))
        return correct.sum() / len(y_true)
    
    def _simulate_sharpe(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Simulate trading strategy and compute Sharpe ratio.
        Trade when prediction exceeds threshold.
        """
        threshold = self.config.execution.min_edge_bps / 10000  # Convert bps to decimal
        
        # Generate signals
        signals = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
        
        # Simulated returns (actual return * signal)
        strategy_returns = pd.Series(y_true.values * signals)
        
        # Filter out no-trade periods
        strategy_returns = strategy_returns[strategy_returns != 0]
        
        if len(strategy_returns) < 2:
            return 0.0
        
        # Calculate Sharpe (assume 1-minute bars, annualize)
        sharpe = calculate_sharpe_ratio(strategy_returns, periods_per_year=252*24*60)
        
        return sharpe
    
    def retrain_if_needed(
        self, 
        df_1m: pd.DataFrame,
        current_metrics: Dict = None
    ) -> Tuple[bool, lgb.Booster, Dict]:
        """
        Decide if retraining is needed and execute.
        
        Args:
            df_1m: Current data
            current_metrics: Metrics from active model
            
        Returns:
            should_retrain: Boolean
            new_model: New model if retrained
            new_metadata: Metadata if retrained
        """
        # Always retrain if no current model
        if current_metrics is None:
            self.logger.info("retraining_no_current_model")
            model, metadata = self.train(df_1m)
            return True, model, metadata
        
        # Check data sufficiency
        if len(df_1m) < self.config.training.min_training_rows:
            self.logger.warning("insufficient_data_for_retraining")
            return False, None, None
        
        # Train candidate model
        self.logger.info("training_candidate_model")
        candidate_model, candidate_metadata = self.train(df_1m)
        
        # Compare metrics
        should_promote = self._should_promote_candidate(
            current_metrics,
            candidate_metadata['metrics']
        )
        
        if should_promote:
            self.logger.info(
                "promoting_candidate",
                current_val_rmse=current_metrics.get('val_rmse'),
                candidate_val_rmse=candidate_metadata['metrics']['val_rmse']
            )
            return True, candidate_model, candidate_metadata
        else:
            self.logger.info("candidate_rejected")
            return False, None, None
    
    def _should_promote_candidate(self, current_metrics: Dict, candidate_metrics: Dict) -> bool:
        """
        Decide if candidate model should be promoted.
        
        Promotion criteria:
        - Better validation RMSE
        - Better directional accuracy
        - Better Sharpe ratio
        """
        current_rmse = current_metrics.get('val_rmse', float('inf'))
        candidate_rmse = candidate_metrics['val_rmse']
        
        current_sharpe = current_metrics.get('val_sharpe', 0.0)
        candidate_sharpe = candidate_metrics['val_sharpe']
        
        # Primary: RMSE improvement
        rmse_improved = candidate_rmse < current_rmse * 0.98  # 2% improvement threshold
        
        # Secondary: Sharpe improvement
        sharpe_improved = candidate_sharpe > current_sharpe * 1.05  # 5% improvement
        
        # Promote if either metric significantly improved
        return rmse_improved or sharpe_improved
