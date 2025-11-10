"""
Utility functions for common operations.
"""
import hashlib
import pandas as pd
import numpy as np
from typing import Any
import json


def hash_dataframe(df: pd.DataFrame) -> str:
    """Generate deterministic hash of DataFrame for dataset versioning."""
    df_string = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(df_string.tobytes()).hexdigest()


def hash_dict(d: dict) -> str:
    """Generate deterministic hash of dictionary."""
    dict_string = json.dumps(d, sort_keys=True)
    return hashlib.sha256(dict_string.encode()).hexdigest()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator."""
    return numerator / denominator if denominator != 0 else default


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252 * 24 * 60) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (default: minute bars)
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve."""
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for distribution shift detection.
    
    Args:
        expected: Expected (baseline) distribution
        actual: Actual (current) distribution
        bins: Number of bins for histogram
        
    Returns:
        PSI value (>0.25 indicates significant shift)
    """
    def scale_range(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(x)
    
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    
    expected_scaled = scale_range(expected, min_val, max_val)
    actual_scaled = scale_range(actual, min_val, max_val)
    
    expected_hist, bin_edges = np.histogram(expected_scaled, bins=bins, range=(0, 1))
    actual_hist, _ = np.histogram(actual_scaled, bins=bins, range=(0, 1))
    
    expected_pct = expected_hist / len(expected)
    actual_pct = actual_hist / len(actual)
    
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi_value


def ensure_dir(path: str):
    """Ensure directory exists, create if not."""
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
