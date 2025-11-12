"""
Simplified feature engineering service using only essential technical indicators:
- EMA (12, 26 periods)
- RSI (14 period)
- VWAP (Volume Weighted Average Price)
- Spread (high-low range)
- Price/Volume momentum
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import json
from pathlib import Path

from src.config import Config
from src.utils import get_logger, ensure_dir


class FeatureService:
    """
    Simplified feature service using only essential technical indicators.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("feature_service")
        
        # Simplified feature configuration
        self.ema_fast = 12
        self.ema_slow = 26
        self.rsi_period = 14
        self.vwap_period = 60  # 1 hour rolling VWAP
        
        self.feature_spec = self._build_feature_spec()
    
    def _build_feature_spec(self) -> dict:
        """Build feature specification."""
        return {
            'ema_fast': self.ema_fast,
            'ema_slow': self.ema_slow,
            'rsi_period': self.rsi_period,
            'vwap_period': self.vwap_period,
            'version': '2.0'
        }
    
    def save_feature_spec(self, path: str):
        """Save feature spec to JSON."""
        ensure_dir(Path(path).parent)
        with open(path, 'w') as f:
            json.dump(self.feature_spec, f, indent=2)
        self.logger.info("feature_spec_saved", path=path)
    
    def _compute_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Compute Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = series.diff()
        
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compute_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
        return vwap
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features from 1-minute OHLCV data."""
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. EMAs
        df['ema_12'] = self._compute_ema(df['close'], self.ema_fast)
        df['ema_26'] = self._compute_ema(df['close'], self.ema_slow)
        df['ema_diff'] = df['ema_12'] - df['ema_26']
        df['ema_diff_pct'] = (df['ema_12'] / df['ema_26'] - 1) * 100
        
        # 2. RSI
        df['rsi'] = self._compute_rsi(df['close'], self.rsi_period)
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]
        
        # 3. VWAP
        df['vwap'] = self._compute_vwap(df, self.vwap_period)
        df['price_to_vwap'] = (df['close'] / df['vwap'] - 1) * 100
        
        # 4. Spread (high-low range)
        df['spread'] = (df['high'] - df['low']) / df['close'] * 100
        df['spread_ma'] = df['spread'].rolling(15).mean()
        
        # 5. Returns (multiple windows)
        df['ret_1m'] = df['close'].pct_change(1) * 100
        df['ret_5m'] = df['close'].pct_change(5) * 100
        df['ret_15m'] = df['close'].pct_change(15) * 100
        
        # 6. Volume momentum
        df['volume_ma_15'] = df['volume'].rolling(15).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_15'] + 1e-10)
        
        # 7. Price momentum (distance from MA)
        df['price_ma_15'] = df['close'].rolling(15).mean()
        df['price_to_ma'] = (df['close'] / df['price_ma_15'] - 1) * 100
        
        # 8. Volatility (std of returns) - key for volatile periods
        df['volatility_15m'] = df['close'].pct_change().rolling(15).std() * 100
        df['volatility_60m'] = df['close'].pct_change().rolling(60).std() * 100
        
        # 9. Price acceleration (rate of change of returns)
        df['acceleration'] = df['ret_1m'].diff()
        
        # 10. Volume-price divergence (volume increasing while price falling = potential reversal)
        df['volume_price_div'] = (df['volume_ratio'] - 1) * (df['ret_5m'])
        
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
        
        df = self._compute_features(df_1m)
        
        # Target: 5-minute forward return
        horizon = self.config.training.prediction_horizon_minutes
        df['target'] = (df['close'].shift(-horizon) / df['close'] - 1) * 100
        
        # Remove rows with NaN in target
        df = df.dropna(subset=['target'])
        
        # Feature columns (19 features now, added volatility & momentum)
        feature_cols = [
            'ema_12', 'ema_26', 'ema_diff', 'ema_diff_pct',
            'rsi', 'rsi_normalized',
            'vwap', 'price_to_vwap',
            'spread', 'spread_ma',
            'ret_1m', 'ret_5m', 'ret_15m',
            'volume_ratio', 'price_to_ma',
            'volatility_15m', 'volatility_60m',  # NEW: volatility features
            'acceleration', 'volume_price_div'   # NEW: momentum features
        ]
        
        # Fill NaN values (forward fill -> backward fill -> zeros)
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
        
        X = df[feature_cols]
        y = df['target']
        
        self.logger.info(
            "train_features_computed",
            num_samples=len(X),
            num_features=len(feature_cols)
        )
        
        return X, y
    
    def compute_live_features(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for live prediction (no target).
        Uses EXACT same features as training.
        
        Args:
            df_1m: Historical 1-minute bars
            
        Returns:
            X_live: Feature matrix for latest timestamp
        """
        if df_1m.empty:
            return pd.DataFrame()
        
        df = self._compute_features(df_1m)
        
        # Feature columns (MUST match training features - 19 features)
        feature_cols = [
            'ema_12', 'ema_26', 'ema_diff', 'ema_diff_pct',
            'rsi', 'rsi_normalized',
            'vwap', 'price_to_vwap',
            'spread', 'spread_ma',
            'ret_1m', 'ret_5m', 'ret_15m',
            'volume_ratio', 'price_to_ma',
            'volatility_15m', 'volatility_60m',  
            'acceleration', 'volume_price_div'   
        ]
        
        # Get latest row with valid features
        df = df.dropna(subset=feature_cols)
        
        if df.empty:
            return pd.DataFrame()
        
        X_live = df[feature_cols].iloc[[-1]]
        
        return X_live
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names (19 features total)."""
        return [
            'ema_12', 'ema_26', 'ema_diff', 'ema_diff_pct',
            'rsi', 'rsi_normalized',
            'vwap', 'price_to_vwap',
            'spread', 'spread_ma',
            'ret_1m', 'ret_5m', 'ret_15m',
            'volume_ratio', 'price_to_ma',
            'volatility_15m', 'volatility_60m',
            'acceleration', 'volume_price_div'
        ]
