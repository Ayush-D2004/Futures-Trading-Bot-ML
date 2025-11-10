"""
Shared data schemas and types used across modules.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    NEW = "NEW"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


@dataclass
class Tick:
    """Raw trade tick from Binance WebSocket."""
    timestamp: datetime
    symbol: str
    price: float
    quantity: float
    side: OrderSide
    trade_id: str


@dataclass
class OHLCV:
    """Aggregated OHLCV bar."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    num_trades: int


@dataclass
class Position:
    """Current position state."""
    symbol: str
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    last_update: datetime


@dataclass
class Order:
    """Order request/response."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    client_order_id: Optional[str] = None
    status: Optional[OrderStatus] = None
    filled_quantity: Optional[float] = None
    filled_price: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class ModelMetadata:
    """Model registry metadata."""
    model_version: str
    training_time: datetime
    dataset_hash: str
    feature_spec: dict
    metrics: dict
    params: dict
    status: Literal["candidate", "staging", "active"]


@dataclass
class Prediction:
    """Model prediction output."""
    timestamp: datetime
    symbol: str
    predicted_return: float
    model_version: str
    features: dict


@dataclass
class Trade:
    """Executed trade record."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    fee: float
    model_version: str
    prediction: float
