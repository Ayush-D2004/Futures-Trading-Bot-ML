"""
Example script to run a backtest on historical data.
"""
import pandas as pd
from datetime import datetime, timedelta

from src.config import load_config
from src.utils import setup_logging
from src.features import FeatureService
from src.model import Trainer, ModelRegistry
from src.backtesting import Backtester


def run_backtest():
    """Run backtest example."""
    print("=" * 60)
    print("TRC Quant Bot - Backtesting")
    print("=" * 60)
    
    # Load config
    config = load_config()
    setup_logging(log_dir=config.monitoring.log_dir)
    
    # Initialize components
    feature_service = FeatureService(config)
    registry = ModelRegistry(config)
    
    print("\n[1/4] Generating synthetic historical data...")
    # Create synthetic data (in production, load from Parquet files)
    dates = pd.date_range(end=datetime.now(), periods=1440, freq='1min')  # 24 hours
    
    df_1m = pd.DataFrame({
        'timestamp': dates,
        'symbol': config.data.symbol,
        'open': 50000 + pd.Series(range(len(dates))).apply(lambda x: x % 200 - 100),
        'high': 50000 + pd.Series(range(len(dates))).apply(lambda x: x % 200),
        'low': 50000 + pd.Series(range(len(dates))).apply(lambda x: x % 200 - 200),
        'close': 50000 + pd.Series(range(len(dates))).apply(lambda x: (x % 200) - 100),
        'volume': 100.0,
        'num_trades': 10
    })
    
    print(f"✓ Generated {len(df_1m)} 1-minute bars")
    
    # Train model on first 12 hours
    print("\n[2/4] Training model on first 12 hours...")
    train_df = df_1m.iloc[:720]  # First 12 hours
    
    trainer = Trainer(config, feature_service)
    model, metadata = trainer.train(train_df)
    
    print(f"✓ Model trained")
    print(f"   Val RMSE: {metadata['metrics']['val_rmse']:.6f}")
    print(f"   Val Sharpe: {metadata['metrics']['val_sharpe']:.3f}")
    
    # Register model
    version = registry.register(model, metadata, status="active")
    print(f"✓ Model registered: {version}")
    
    # Run backtest on second 12 hours (out-of-sample)
    print("\n[3/4] Running backtest on out-of-sample data...")
    test_df = df_1m.iloc[720:]  # Second 12 hours
    
    backtester = Backtester(config, feature_service, registry)
    results = backtester.run(test_df, model_version="active")
    
    # Display results
    print("\n[4/4] Backtest Results:")
    print("=" * 60)
    print(f"Total PnL:        ${results['total_pnl']:,.2f}")
    print(f"Total Return:     {results['total_return']:.2%}")
    print(f"Num Trades:       {results['num_trades']}")
    print(f"Win Rate:         {results['win_rate']:.2%}")
    print(f"Avg Win:          ${results['avg_win']:,.2f}")
    print(f"Avg Loss:         ${results['avg_loss']:,.2f}")
    print(f"Sharpe Ratio:     {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:     {results['max_drawdown']:.2%}")
    print("=" * 60)
    
    # Save results
    backtester.save_results(results)
    print("\n✓ Results saved to backtests/")
    
    # Try to plot (requires matplotlib)
    try:
        backtester.plot_results()
        print("✓ Plots saved to backtests/")
    except:
        print("⚠ Plotting skipped (matplotlib not installed)")
    
    print("\n✅ Backtest completed successfully!")


if __name__ == "__main__":
    run_backtest()
