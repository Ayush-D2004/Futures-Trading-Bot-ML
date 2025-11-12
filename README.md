# Crypto ML Trading Bot

A production-ready algorithmic trading bot for cryptocurrency futures using machine learning predictions with rolling retraining.

## 🎯 Core Strategy

### Trading Logic
The bot uses **LightGBM regression** to predict 10-minute forward returns along with several technical indicators, executing trades when predictions exceed cost thresholds.

**Position Management:**
- **Entry**: Opens LONG/SHORT when prediction exceeds 8 bps (0.08%) threshold
- **Hold**: Maintains position across model retraining cycles  
- **Exit**: Closes position on signal reversal (4.5 bps), stop-loss (1%), or take-profit (3%)
- **Cooldown**: 3-minute wait between trades.

---

## 🏗️ System Architecture

![System Architecture](architecture.png)

---

## 📊 Data Pipeline

### Data Flow
```
Binance WebSocket (@kline_1m) → 1-minute bars → Feature Engineering → LightGBM Model → Predictions
```

**Live Data:**
1. WebSocket streams completed 1-minute klines from Binance
2. Buffer maintains rolling window (48-72 hours) in memory
3. Features computed on the fly from OHLCV data
4. Model predicts once per minute when new bar completes

**Feature Engineering (19 Features):**
- **Price**: EMA(12,26,60), VWAP, price momentum vs MA
- **Momentum**: Returns (1m, 5m, 15m), RSI, acceleration
- **Volatility**: 15-min & 60-min std, spread, spread MA
- **Volume**: Volume ratio, volume-price divergence

---

## Model Retraining

### Live Trading
```
Training Data: 48 hours of 1-minute bars (2,880 samples)
Retraining: Every 15 minutes
Window: Rolling 48-hour window
```

**Process:**
1. Fetch latest 48 hours from Binance
2. Compute features for all bars
3. Train LightGBM (80/20 train/val split)
4. Validate metrics (RMSE, Sharpe, accuracy)
5. Register model with versioning
6. Swap to new model if validation passes
7. Repeat every 15 minutes

### Backtesting
```
Training: First 48 hours of dataset
Testing: Remaining 24 hours
Retraining: Every 60 minutes during test
```

**Process:**
1. Load 72 hours historical data (48h train + 24h test)
2. Train initial model on first 48 hours
3. Test on remaining 24 hours with minute-by-minute execution
4. Retrain every hour with rolling 48-hour window
5. Track equity, trades, and PnL throughout

## Model Configuration:
```yaml
LightGBM:
  num_leaves: 64        # Complex pattern capture
  max_depth: 8          # Deep learning
  n_estimators: 1000    # Ensemble strength
  learning_rate: 0.03   # Stable convergence
  subsample: 0.8        # Regularization
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
python -m venv quant
quant\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Binance API

1. Get API credentials from https://www.binance.com/en/my/settings/api-management
2. Copy `.env.example` to `.env`
3. Add your credentials:
```bash
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```

### 3. Run Backtest

```powershell
python run_backtest.py
```

Output:
- Trade log with entry/exit prices
- Win rate and PnL statistics
- Sharpe ratio and max drawdown
- Results saved to `backtests/`

### 4. Run Live Bot
Switch between Paper Trading as needed from config file.

```powershell
python main.py
```

Features:
- Real-time 1-minute predictions
- Automatic position management
- Rolling model retraining every 15 minutes
- Paper trading mode (default, safe)

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
data:
  symbol: "BTCUSDT"               # Trading pair
  rolling_window_hours: 12        # Training data window
  
training:
  train_freq_minutes: 15          # Retraining frequency
  prediction_horizon_minutes: 5   # Prediction target
  
retraining:
  psi_threshold: 0.25             # Drift detection threshold
  sharpe_threshold: 0.4           # Performance threshold
  shadow_mode_minutes: 5          # Shadow mode duration
  
execution:
  paper_trading: true             # Set false for live trading
  base_capital: 10000.0           # Starting capital
  max_daily_drawdown_pct: 3.0     # Kill switch trigger
```

## 🔬 Retraining Pipeline

The bot retrains when:

1. **Scheduled**: Every 15 minutes (configurable)
2. **Feature Drift**: PSI > 0.25 on any feature
3. **Performance Degradation**: Rolling Sharpe < 0.4


## 📚 References

- Binance API: https://binance-docs.github.io/apidocs/
- LightGBM: https://lightgbm.readthedocs.io/

## 🤝 Contributing

This is a research project showcasing ML Ops and quant finance best practices.
