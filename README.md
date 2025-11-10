# TRC Quant Trading Bot

Advanced cryptocurrency trading bot with LightGBM ML model, real-time data pipeline, and automated retraining.

## 🎯 Features

- **Real-time Live Market Data**: Always uses Binance mainnet for real market data
- **Multi-resolution Aggregation**: 1-second and 1-minute OHLCV bars
- **Feature Engineering**: 23 technical indicators with deterministic computation
- **LightGBM Model**: Fast regression model predicting 5-minute forward returns
- **Automated Retraining**: 15-minute cadence with drift detection (PSI) and performance monitoring
- **Shadow Mode**: Safe model deployment with validation before activation
- **Risk Management**: Stop-loss (2%), take-profit (5%), peak PnL drawdown (20%), daily limits
- **Execution Engine**: Paper trading or live execution modes
- **Monitoring**: Metrics logging, PSI tracking, trade logging to Parquet

## 📁 Project Structure

```
Quant Bot/
├── src/
│   ├── data/               # Data pipeline (Ingestor, Aggregator, BufferManager)
│   ├── features/           # Feature engineering service
│   ├── model/              # Trainer and ModelRegistry
│   ├── prediction/         # Predictor service
│   ├── execution/          # ExecutionEngine with Binance API
│   ├── monitoring/         # Monitor for metrics and alerts
│   ├── utils/              # Helper functions and logging
│   ├── config.py           # Pydantic configuration models
│   ├── schemas.py          # Data schemas
│   └── orchestrator.py     # Main bot orchestration
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Binance API

1. Get API credentials from: https://www.binance.com/en/my/settings/api-management
2. Copy `.env.example` to `.env`:
3. Edit `.env` and add your credentials:


### 3. Run the Bot

```powershell
python -m src.orchestrator
```

The bot will:
- Connect to Binance WebSocket for BTCUSDT
- Start ingesting and aggregating data
- Wait for 12 hours of data (configurable)
- Train initial model
- Start making predictions every second
- Retrain every 15 minutes

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
data:
  symbol: "BTCUSDT"              # Trading pair
  rolling_window_hours: 12        # Training data window
  
training:
  train_freq_minutes: 15          # Retraining frequency
  prediction_horizon_minutes: 5   # Prediction target
  
retraining:
  psi_threshold: 0.25            # Drift detection threshold
  sharpe_threshold: 0.4          # Performance threshold
  shadow_mode_minutes: 5         # Shadow mode duration
  
execution:
  paper_trading: true            # Set false for live trading
  base_capital: 10000.0          # Starting capital
  max_daily_drawdown_pct: 3.0    # Kill switch trigger
```

## 🔬 Retraining Pipeline

The bot retrains when:

1. **Scheduled**: Every 15 minutes (configurable)
2. **Feature Drift**: PSI > 0.25 on any feature
3. **Performance Degradation**: Rolling Sharpe < 0.4


## 🛡️ Risk Controls

- **Position Limits**: Max position size per trade
- **Daily Drawdown**: Kill switch at 3% daily loss
- **Cost Model**: Fees (0.04%) + Slippage (0.03%)
- **Min Edge**: Only trade if expected return > 0.5%

## 📈 Performance Metrics

Tracked metrics:
- RMSE (train/validation)
- Directional accuracy
- Simulated Sharpe ratio
- Feature PSI scores
- Daily PnL
- Position state

## 📚 References

- Binance API: https://binance-docs.github.io/apidocs/
- LightGBM: https://lightgbm.readthedocs.io/

## 🤝 Contributing

This is a research project showcasing ML Ops and quant finance best practices.
