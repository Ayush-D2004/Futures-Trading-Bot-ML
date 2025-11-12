"""
Configuration models using Pydantic for type safety and validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
import yaml
from pathlib import Path


class DataConfig(BaseModel):
    symbol: str
    rolling_window_hours: int
    tick_buffer_size: int
    parquet_roll_minutes: int
    raw_data_dir: str
    processed_data_dir: str
    prediction_interval_seconds: int = 60  # Default: predict once per minute


class BinanceConfig(BaseModel):
    use_testnet: bool
    mainnet_ws_url: str
    api_key_env: str
    api_secret_env: str


class FuturesConfig(BaseModel):
    enabled: bool
    leverage: int = Field(ge=1, le=125)  # Binance allows 1-125x
    margin_type: Literal["CROSSED", "ISOLATED"]
    position_mode: Literal["ONE_WAY", "HEDGE"]


class LightGBMParams(BaseModel):
    num_leaves: int
    max_depth: int
    learning_rate: float
    n_estimators: int
    boosting_type: str
    objective: str
    metric: str
    verbosity: int
    random_state: int
    early_stopping_rounds: int


class TrainingConfig(BaseModel):
    model_type: Literal["lightgbm", "xgboost"]
    train_freq_minutes: int
    prediction_horizon_minutes: int
    min_training_rows: int
    validation_split: float
    lightgbm_params: LightGBMParams


class FeaturesConfig(BaseModel):
    rolling_windows: List[int]
    ema_periods: List[int]
    rsi_period: int
    volatility_window: int


class RetrainingConfig(BaseModel):
    scheduled_minutes: int
    psi_threshold: float
    sharpe_threshold: float
    shadow_mode_minutes: int
    min_shadow_predictions: int
    max_model_age_minutes: int


class ExecutionConfig(BaseModel):
    paper_trading: bool
    order_type: Literal["LIMIT", "MARKET"]
    base_capital: float
    position_size_fraction: float
    target_volatility: float
    max_position_usd: float
    max_daily_drawdown_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    peak_pnl_drawdown_pct: float
    fee_rate: float
    slippage_rate: float
    latency_seconds: float
    min_edge_bps: float


class MonitoringConfig(BaseModel):
    log_dir: str
    metrics_interval_seconds: int
    alert_channels: List[str]


class ModelRegistryConfig(BaseModel):
    models_dir: str
    keep_last_n_models: int


class DatabaseConfig(BaseModel):
    sqlite_path: str


class BacktestingConfig(BaseModel):
    initial_capital: float
    results_dir: str
    train_hours: int = 12
    test_start_offset_hours: int = 24
    retrain_interval_minutes: int = 60


class Config(BaseModel):
    data: DataConfig
    binance: BinanceConfig
    futures: FuturesConfig
    training: TrainingConfig
    features: FeaturesConfig
    retraining: RetrainingConfig
    execution: ExecutionConfig
    monitoring: MonitoringConfig
    model_registry: ModelRegistryConfig
    database: DatabaseConfig
    backtesting: BacktestingConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load and validate configuration from YAML file."""
    return Config.from_yaml(config_path)
