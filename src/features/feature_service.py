"""
Feature engineering service for training and live predictions.
Ensures deterministic feature computation between training and serving.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import json
from pathlib import Path

from src.config import Config
from src.utils import get_logger, ensure_dir


class FeatureService:
    """
    Computes features for both training (1m bars) and execution (1s micro-features).
    Feature computation must be identical between training and serving to avoid leakage.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("feature_service")
        
        # Feature configuration
        self.rolling_windows = config.features.rolling_windows
        self.ema_periods = config.features.ema_periods
        self.rsi_period = config.features.rsi_period
        self.volatility_window = config.features.volatility_window
        
        self.feature_spec = self._build_feature_spec()
    
    def _build_feature_spec(self) -> dict:
        """Build feature specification for reproducibility."""
        return {
            'rolling_windows': self.rolling_windows,
            'ema_periods': self.ema_periods,
            'rsi_period': self.rsi_period,
            'volatility_window': self.volatility_window,
            'version': '1.0'
        }
    
    def save_feature_spec(self, path: str):
        """Save feature spec to JSON."""
        ensure_dir(Path(path).parent)
        with open(path, 'w') as f:
            json.dump(self.feature_spec, f, indent=2)
        self.logger.info("feature_spec_saved", path=path)
    
    def _compute_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns."""
        df = df.copy()
        df['log_ret_1m'] = np.log(df['close'] / df['close'].shift(1))
        return df
    
    def _compute_rolling_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling returns over multiple windows."""
        df = df.copy()
        
        for window in self.rolling_windows:
            df[f'ret_{window}m'] = df['log_ret_1m'].rolling(window).sum()
        
        return df
    
    def _compute_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute EMA-based features."""
        df = df.copy()
        
        for period in self.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # EMA differences
        if len(self.ema_periods) >= 2:
            df['ema_diff_12_26'] = df['ema_12'] - df['ema_26']
            df['ema_diff_12_60'] = df['ema_12'] - df['ema_60']
        
        return df
    
    def _compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = df['close'].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute realized volatility."""
        df = df.copy()
        df['volatility'] = df['log_ret_1m'].rolling(self.volatility_window).std()
        return df
    
    def _compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features."""
        df = df.copy()
        
        # Volume rolling mean
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_15'] = df['volume'].rolling(15).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_ma_15']
        
        return df
    
    def _compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price-based features."""
        df = df.copy()
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Close position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['close_position'] = df['close_position'].fillna(0.5)
        
        return df
    
    def _compute_drawdown(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Compute rolling drawdown."""
        df = df.copy()
        rolling_max = df['close'].rolling(window).max()
        df['drawdown'] = (df['close'] / rolling_max) - 1
        return df
    
    def _compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-based cyclical features."""
        df = df.copy()
        
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        
        df = df.drop(['hour', 'minute'], axis=1)
        
        return df
    
    def compute_train_features(self, df_1m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute training features from 1-minute bars.
        
        Args:
            df_1m: DataFrame with 1-minute OHLCV bars
            
        Returns:
            X: Feature matrix
            y: Target (5-minute forward return)
        """
        if df_1m.empty:
            return pd.DataFrame(), pd.Series()
        
        df = df_1m.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Price features
        df = self._compute_log_returns(df)
        df = self._compute_rolling_returns(df)
        df = self._compute_ema_features(df)
        df['rsi'] = self._compute_rsi(df, self.rsi_period)
        df = self._compute_volatility(df)
        df = self._compute_drawdown(df)
        df = self._compute_volume_features(df)
        df = self._compute_price_features(df)
        df = self._compute_time_features(df)
        
        # Target: 5-minute forward return (strict forward indexing)
        horizon = self.config.training.prediction_horizon_minutes
        df['target_5m'] = (df['close'].shift(-horizon) / df['close']) - 1
        
        # Remove rows with NaN in target
        df = df.dropna(subset=['target_5m'])
        
        # Feature columns (exclude metadata and target)
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 
                       'volume', 'num_trades', 'target_5m']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Drop rows with NaN in features
        df = df.dropna(subset=feature_cols)
        
        X = df[feature_cols]
        y = df['target_5m']
        
        self.logger.debug(
            "train_features_computed",
            num_samples=len(X),
            num_features=len(feature_cols),
            features=feature_cols
        )
        
        return X, y
    
    def compute_live_features(self, df_1m: pd.DataFrame, df_1s_recent: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compute features for live prediction (no target).
        
        Args:
            df_1m: Historical 1-minute bars
            df_1s_recent: Recent 1-second bars for micro-features (optional for MVP)
            
        Returns:
            X_live: Feature matrix for latest timestamp
        """
        if df_1m.empty:
            return pd.DataFrame()
        
        df = df_1m.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Compute same features as training
        df = self._compute_log_returns(df)
        df = self._compute_rolling_returns(df)
        df = self._compute_ema_features(df)
        df['rsi'] = self._compute_rsi(df, self.rsi_period)
        df = self._compute_volatility(df)
        df = self._compute_drawdown(df)
        df = self._compute_volume_features(df)
        df = self._compute_price_features(df)
        df = self._compute_time_features(df)
        
        # Micro-features from 1s data (optional)
        if df_1s_recent is not None and not df_1s_recent.empty:
            df = self._compute_micro_features(df, df_1s_recent)
        
        # Feature columns
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 
                       'volume', 'num_trades']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Get latest row
        df = df.dropna(subset=feature_cols)
        
        if df.empty:
            return pd.DataFrame()
        
        X_live = df[feature_cols].iloc[[-1]]
        
        return X_live
    
    def _compute_micro_features(self, df_1m: pd.DataFrame, df_1s: pd.DataFrame) -> pd.DataFrame:
        """
        Compute execution micro-features from 1-second data.
        (Placeholder for future orderbook features)
        """
        df = df_1m.copy()
        
        if df_1s.empty:
            return df
        
        # Recent volume spike
        recent_volume = df_1s['volume'].tail(10).sum()
        median_volume = df_1s['volume'].median()
        
        df['volume_spike'] = recent_volume / median_volume if median_volume > 0 else 1.0
        
        # Short-term price slope (last 10 seconds)
        if len(df_1s) >= 10:
            recent_prices = df_1s['close'].tail(10).values
            x = np.arange(len(recent_prices))
            slope = np.polyfit(x, recent_prices, 1)[0]
            df['price_slope_10s'] = slope
        else:
            df['price_slope_10s'] = 0.0
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for model training."""
        # Create dummy dataframe to extract feature names
        dummy_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'symbol': 'BTCUSDT',
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50000.0,
            'volume': 100.0,
            'num_trades': 10
        })
        
        X, _ = self.compute_train_features(dummy_df)
        return list(X.columns)
