"""
Monitoring service for metrics tracking, PSI calculation, and alerting.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path

from src.config import Config
from src.utils import get_logger, calculate_psi, ensure_dir
from src.schemas import Prediction


class Monitor:
    """
    Monitors model performance, feature drift, and trading metrics.
    Emits alerts when thresholds are breached.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("monitor")
        
        # Metrics storage
        self.metrics_history = []
        self.psi_history = []
        
        # Baseline features for PSI calculation
        self.baseline_features: Dict[str, np.ndarray] = {}
        
        # Trade history
        self.trades_file = Path(config.monitoring.log_dir) / "trades.parquet"
        ensure_dir(Path(config.monitoring.log_dir))
    
    def set_baseline_features(self, X: pd.DataFrame):
        """Set baseline feature distributions for PSI calculation."""
        for col in X.columns:
            self.baseline_features[col] = X[col].dropna().values
        
        self.logger.info(
            "baseline_features_set",
            num_features=len(self.baseline_features)
        )
    
    def calculate_feature_psi(self, X_current: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate PSI for each feature against baseline.
        
        Returns:
            dict: Feature name -> PSI value
        """
        if not self.baseline_features:
            self.logger.warning("no_baseline_features")
            return {}
        
        psi_scores = {}
        
        for col in X_current.columns:
            if col in self.baseline_features:
                baseline = self.baseline_features[col]
                current = X_current[col].dropna().values
                
                if len(current) > 10:  # Need sufficient samples
                    psi = calculate_psi(baseline, current)
                    psi_scores[col] = psi
        
        return psi_scores
    
    def check_drift(self, X_current: pd.DataFrame) -> tuple[bool, Dict]:
        """
        Check for feature drift using PSI.
        
        Returns:
            drift_detected: Boolean
            psi_scores: Dict of PSI scores
        """
        psi_scores = self.calculate_feature_psi(X_current)
        
        if not psi_scores:
            return False, {}
        
        max_psi = max(psi_scores.values())
        threshold = self.config.retraining.psi_threshold
        
        drift_detected = max_psi > threshold
        
        if drift_detected:
            # Find top drifted features
            top_drift = sorted(
                psi_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            self.logger.warning(
                "feature_drift_detected",
                max_psi=max_psi,
                threshold=threshold,
                top_drifted_features=top_drift
            )
        
        # Store PSI history
        self.psi_history.append({
            'timestamp': datetime.now(),
            'max_psi': max_psi,
            'psi_scores': psi_scores
        })
        
        return drift_detected, psi_scores
    
    def log_trade(self, trade_data: Dict):
        """Log executed trade to persistent storage."""
        try:
            # Load existing trades
            if self.trades_file.exists():
                df_trades = pd.read_parquet(self.trades_file)
                df_new = pd.DataFrame([trade_data])
                df_trades = pd.concat([df_trades, df_new], ignore_index=True)
            else:
                df_trades = pd.DataFrame([trade_data])
            
            # Save
            df_trades.to_parquet(self.trades_file, index=False)
            
            self.logger.info("trade_logged", trade=trade_data)
            
        except Exception as e:
            self.logger.error("trade_log_error", error=str(e))
    
    def calculate_rolling_sharpe(self, lookback_trades: int = 200) -> float:
        """Calculate Sharpe ratio on recent trades."""
        if not self.trades_file.exists():
            return 0.0
        
        try:
            df_trades = pd.read_parquet(self.trades_file)
            
            if len(df_trades) < 10:
                return 0.0
            
            # Get recent trades
            df_recent = df_trades.tail(lookback_trades)
            
            if 'pnl' not in df_recent.columns:
                return 0.0
            
            returns = df_recent['pnl'].values
            
            if len(returns) < 2:
                return 0.0
            
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            
            if std_ret == 0:
                return 0.0
            
            # Annualized Sharpe (assuming 1 trade per 5 minutes on average)
            trades_per_year = 365 * 24 * 12  # 12 trades per hour
            sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)
            
            return sharpe
            
        except Exception as e:
            self.logger.error("sharpe_calculation_error", error=str(e))
            return 0.0
    
    def check_performance_degradation(self) -> bool:
        """Check if trading performance has degraded below threshold."""
        sharpe = self.calculate_rolling_sharpe()
        threshold = self.config.retraining.sharpe_threshold
        
        if sharpe < threshold:
            self.logger.warning(
                "performance_degradation_detected",
                sharpe=sharpe,
                threshold=threshold
            )
            return True
        
        return False
    
    def log_metrics(self, metrics: Dict):
        """Log system metrics."""
        metrics['timestamp'] = datetime.now()
        self.metrics_history.append(metrics)
        
        # Keep limited history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        self.logger.info("metrics_logged", metrics=metrics)
    
    def emit_alert(self, alert_type: str, message: str, data: Dict = None):
        """Emit alert through configured channels."""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'data': data
        }
        
        # File logging
        if 'file' in self.config.monitoring.alert_channels:
            alert_file = Path(self.config.monitoring.log_dir) / "alerts.log"
            with open(alert_file, 'a') as f:
                f.write(f"{alert}\n")
        
        self.logger.warning("alert_emitted", alert=alert)
    
    def get_system_status(self) -> Dict:
        """Get current system status summary."""
        status = {
            'timestamp': datetime.now(),
            'metrics_count': len(self.metrics_history),
            'psi_checks': len(self.psi_history),
            'rolling_sharpe': self.calculate_rolling_sharpe()
        }
        
        if self.psi_history:
            latest_psi = self.psi_history[-1]
            status['latest_max_psi'] = latest_psi['max_psi']
        
        return status
