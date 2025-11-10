"""
Test components with REAL live market data from Binance.
No synthetic data - only actual market ticks.
"""
import asyncio
import time
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.data import BinanceIngestor, Aggregator, BufferManager
from src.features import FeatureService
from src.model import Trainer, ModelRegistry
from src.utils import setup_logging


async def collect_live_data(config, duration_minutes=15):
    """Collect real live data from Binance for specified duration."""
    print(f"\n📡 Connecting to Binance WebSocket for {config.data.symbol}...")
    print(f"⏱️  Will collect data for {duration_minutes} minutes...")
    print("   (Press Ctrl+C to stop early)\n")
    
    buffer_manager = BufferManager(config)
    aggregator = Aggregator(tolerance_ms=200)
    
    tick_count = 0
    start_time = time.time()
    
    async def on_tick(tick):
        nonlocal tick_count
        tick_count += 1
        
        # Process tick through aggregator
        completed = aggregator.add_tick(tick)
        
        # Store completed bars in buffer
        if completed['1s']:
            bar = completed['1s']
            buffer_manager.add_bar_1s({
                'timestamp': bar.timestamp,
                'symbol': bar.symbol,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'num_trades': bar.num_trades
            })
        if completed['1m']:
            bar = completed['1m']
            buffer_manager.add_bar_1m({
                'timestamp': bar.timestamp,
                'symbol': bar.symbol,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'num_trades': bar.num_trades
            })
        
        # Progress update every 100 ticks
        if tick_count % 100 == 0:
            elapsed = time.time() - start_time
            bars_1s = buffer_manager.get_bar_count('1s')
            bars_1m = buffer_manager.get_bar_count('1m')
            print(f"   ✓ {tick_count} ticks | {bars_1s} 1s bars | {bars_1m} 1m bars | {elapsed:.1f}s elapsed")
    
    ingestor = BinanceIngestor(config, on_tick_callback=on_tick)
    
    try:
        # Run for specified duration
        task = asyncio.create_task(ingestor.start())
        await asyncio.sleep(duration_minutes * 60)
        await ingestor.stop()
        
    except KeyboardInterrupt:
        print("\n⚠️  Stopping data collection...")
        await ingestor.stop()
    
    return buffer_manager, tick_count


async def main():
    print("=" * 60)
    print("TRC Quant Bot - LIVE DATA Test")
    print("=" * 60)
    
    # Load config
    print("\n[1/5] Loading configuration...")
    config = load_config()
    print(f"✓ Config loaded: Symbol={config.data.symbol}")
    
    # Clear old database
    print("\n[2/5] Clearing old synthetic data...")
    db_path = Path(config.database.sqlite_path)
    if db_path.exists():
        db_path.unlink()
        print(f"✓ Deleted old database: {db_path}")
    else:
        print("✓ No old database found")
    
    # Setup logging
    print("\n[3/5] Setting up logging...")
    logger = setup_logging(log_dir=config.monitoring.log_dir, console_level="WARNING")
    print("✓ Logging configured")
    
    # Collect live data
    print("\n[4/5] Collecting LIVE market data...")
    print(f"   Symbol: {config.data.symbol}")
    print(f"   Source: Binance Mainnet (REAL market)")
    
    buffer_manager, tick_count = await collect_live_data(config, duration_minutes=15)
    
    bars_1s = buffer_manager.get_bar_count('1s')
    bars_1m = buffer_manager.get_bar_count('1m')
    
    print(f"\n✓ Data collection complete!")
    print(f"   Total ticks: {tick_count:,}")
    print(f"   1-second bars: {bars_1s}")
    print(f"   1-minute bars: {bars_1m}")
    
    # Check if we have enough data
    if bars_1m < 100:
        print("\n⚠️  Not enough 1-minute bars for training (need ~100+)")
        print("   Recommendation: Let the bot run longer to collect more data")
        return
    
    # Train on real data
    print("\n[5/5] Training model on REAL market data...")
    
    feature_service = FeatureService(config)
    trainer = Trainer(config, feature_service)
    registry = ModelRegistry(config)
    
    df_1m = buffer_manager.get_bars('1m', lookback_minutes=bars_1m)
    
    print(f"   Training with {len(df_1m)} bars from live market...")
    
    try:
        model, metadata = trainer.train(df_1m)
        
        print(f"\n✓ Model trained on REAL data!")
        print(f"   Train RMSE: {metadata['metrics']['train_rmse']:.6f}")
        print(f"   Val RMSE: {metadata['metrics']['val_rmse']:.6f}")
        print(f"   Directional Accuracy: {metadata['metrics']['direction_accuracy']:.2f}%")
        print(f"   Val Sharpe: {metadata['metrics'].get('val_sharpe', 'N/A')}")
        
        # Register model
        model_version = registry.register(
            model=model,
            metadata=metadata,
            status='staging'
        )
        
        print(f"✓ Model registered: {model_version}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        print("   This is normal if there's not enough data variation")
        return
    
    print("\n" + "=" * 60)
    print("✅ LIVE DATA test complete!")
    print("=" * 60)
    print("\nThis bot uses ONLY real market data.")
    print("Ready to run the full bot with:")
    print("  python -m src.orchestrator")


if __name__ == "__main__":
    asyncio.run(main())
