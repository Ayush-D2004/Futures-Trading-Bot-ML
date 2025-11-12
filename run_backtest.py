import pandas as pd
import asyncio
from datetime import datetime, timedelta
from binance.client import Client
import os

from src.config import load_config
from src.utils import setup_logging
from src.features import FeatureService
from src.model import Trainer, ModelRegistry
from src.backtesting import Backtester


def fetch_binance_historical_data(symbol: str, hours: int = 27) -> pd.DataFrame:
    """
    Fetch real historical data from Binance.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        hours: Hours of historical data to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"   Fetching {hours} hours of real market data from Binance...")
    
    # Initialize Binance client
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    client = Client(api_key, api_secret)
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    # Fetch 1-minute klines
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1MINUTE,
        start_str=start_time.strftime('%Y-%m-%d %H:%M:%S'),
        end_str=end_time.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['trades'] = df['trades'].astype(int)
    
    # Keep only required columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']]
    df.rename(columns={'trades': 'num_trades'}, inplace=True)
    
    print(f"   ✓ Fetched {len(df)} real 1-minute bars")
    print(f"   Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    
    return df


def run_backtest():
    """Run backtest with rolling retraining."""
    print("=" * 60)
    print("Quant Bot - Backtesting with Rolling Retraining")
    print("=" * 60)
    
    # Load config
    config = load_config()
    setup_logging(log_dir=config.monitoring.log_dir)
    
    feature_service = FeatureService(config)
    registry = ModelRegistry(config)
    
    print("\n[1/3] Fetching REAL historical data from Binance...")
    train_hours = config.backtesting.train_hours
    test_hours = config.backtesting.test_start_offset_hours - train_hours
    total_hours = config.backtesting.test_start_offset_hours
    
    df_1m = fetch_binance_historical_data(config.data.symbol, hours=total_hours)
    df_1m['symbol'] = config.data.symbol
    
    print(f"✓ Loaded {len(df_1m)} real 1-minute bars")
    print(f"   Train period: First {train_hours} hours (~{train_hours * 60} bars)")
    print(f"   Test period: Remaining {test_hours} hours (~{test_hours * 60} bars)")
    print(f"   Retraining: Every 1 hour during test")
    
    print(f"\n[2/3] Trading Configuration:")
    print(f"   Fee rate:         {config.execution.fee_rate:.4f} ({config.execution.fee_rate*10000:.1f} bps)")
    print(f"   Slippage rate:    {config.execution.slippage_rate:.4f} ({config.execution.slippage_rate*10000:.1f} bps)")
    print(f"   Min edge:         {config.execution.min_edge_bps} bps")
    total_required = (config.execution.fee_rate + config.execution.slippage_rate) * 10000 + config.execution.min_edge_bps
    print(f"   Total required:   {total_required:.1f} bps ({total_required/100:.2f}%)")
    print(f"   Stop-loss:        {config.execution.stop_loss_pct}%")
    print(f"   Take-profit:      {config.execution.take_profit_pct}%")
    
    print(f"\n[3/3] Running backtest with rolling retraining...")
    backtester = Backtester(config, feature_service, registry)
    results = backtester.run(df_1m)
    
    print("\n[4/4] Backtest Results:")
    print("=" * 60)
    
    if 'error' in results:
        print(f"⚠️  {results['error']}")
        print("   (Model predictions may not have exceeded edge threshold)")
    else:
        print(f"Total PnL:        ${results['total_pnl']:,.2f}")
        print(f"Total Return:     {results['total_return']:.2%}")
        print(f"Num Trades:       {results['num_trades']}")
        print(f"Win Rate:         {results['win_rate']:.2%}")
        print(f"Avg Win:          ${results['avg_win']:,.2f}")
        print(f"Avg Loss:         ${results['avg_loss']:,.2f}")
        print(f"Sharpe Ratio:     {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:     {results['max_drawdown']:.2%}")
    
    print("=" * 60)
    
    if 'error' not in results:
        backtester.save_results(results)
        print("\n✓ Results saved to backtests/")
        
        try:
            backtester.plot_results()
            print("✓ Plots saved to backtests/")
        except:
            print("⚠ Plotting skipped (matplotlib not installed)")
    
    print("\n✅ Backtest completed!")


if __name__ == "__main__":
    run_backtest()
