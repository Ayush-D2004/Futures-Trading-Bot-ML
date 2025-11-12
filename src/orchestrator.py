"""
Main orchestration layer - coordinates all components.
"""
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import signal
import sys
from dotenv import load_dotenv
import pandas as pd

from src.config import load_config
from src.utils import setup_logging
from src.data import BinanceIngestor, Aggregator, BufferManager
from src.features import FeatureService
from src.model import Trainer, ModelRegistry
from src.prediction import Predictor
from src.execution import ExecutionEngine
from src.monitoring import Monitor
from src.schemas import OrderSide, OrderType


class TradingBot:
    """
    Main trading bot orchestrator.
    Coordinates data ingestion, aggregation, prediction, execution, and retraining.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load environment variables from .env file
        load_dotenv()
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(
            log_dir=self.config.monitoring.log_dir,
            level="INFO",
            console_level="INFO"  # Show activity on console
        )
        
        self.logger.info("initializing_trading_bot")
        
        # Initialize components
        self.buffer_manager = BufferManager(self.config)
        self.aggregator = Aggregator()
        self.feature_service = FeatureService(self.config)
        self.model_registry = ModelRegistry(self.config)
        self.trainer = Trainer(self.config, self.feature_service)
        self.predictor = Predictor(self.config, self.model_registry, self.feature_service)
        self.execution_engine = ExecutionEngine(self.config)
        self.monitor = Monitor(self.config)
        
        # Ingestor with callback
        self.ingestor = BinanceIngestor(self.config, on_kline_callback=self._on_kline)
        
        # Scheduler for periodic tasks
        self.scheduler = AsyncIOScheduler()
        
        # State
        self.running = False
        self.last_retrain_time = None
        self.shadow_mode = False
        self.shadow_predictions = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def _on_kline(self, bar):
        """Callback for each completed 1-minute bar."""
        # Add bar to buffer
        self.buffer_manager.add_bar_1m(bar)
        
        # Generate prediction once per minute (when new bar completes)
        await self._generate_prediction()
    
    async def _generate_prediction(self):
        """Generate prediction and make trading decision (once per minute)."""
        # Get 1-minute data
        df_1m = self.buffer_manager.get_rolling_df(
            self.config.data.symbol,
            freq='1m'
        )
        
        if df_1m.empty:
            return
        
        # Reload model if updated
        self.predictor.reload_model_if_updated()
        
        # Generate prediction from 1-minute data only
        prediction = self.predictor.predict(df_1m)
        
        if prediction is None:
            return
        
        # Shadow mode: collect predictions without trading
        if self.shadow_mode:
            self.shadow_predictions.append(prediction)
            self.logger.info("shadow_prediction", prediction=prediction.predicted_return)
            return
        
        # Get current price
        current_price = df_1m.iloc[-1]['close']
        
        # New Strategy: Hold position, close only on signal reversal or risk controls
        # Check risk controls on existing position
        if self.execution_engine.has_position():
            # Check stop-loss
            if self.execution_engine.check_stop_loss(current_price):
                self.execution_engine.close_position(current_price, reason="stop_loss")
                self.logger.info("position_closed_stop_loss")
                return
            
            # Check take-profit
            if self.execution_engine.check_take_profit(current_price):
                self.execution_engine.close_position(current_price, reason="take_profit")
                self.logger.info("position_closed_take_profit")
                return
            
            # Check for signal reversal
            if self.execution_engine.should_reverse_position(prediction.predicted_return):
                self.execution_engine.close_position(current_price, reason="signal_reversal")
                self.logger.info("position_closed_signal_reversal",
                                prediction=prediction.predicted_return)
                # Don't return - may open opposite position below
        
        # Make trading decision (open new position or hold existing)
        action = self.execution_engine.decide_action(
            prediction=prediction.predicted_return,
            current_price=current_price
        )
        
        if action == "OPEN_LONG" or action == "OPEN_SHORT":
            # Open new position
            self.execution_engine.open_position(
                side="LONG" if action == "OPEN_LONG" else "SHORT",
                price=current_price,
                prediction=prediction.predicted_return
            )
            self.logger.info("position_opened",
                           side=action,
                           prediction=prediction.predicted_return)
    
    def _import_numpy(self):
        """Import numpy for shadow mode evaluation."""
        import numpy as np
        return np
    
    async def _retrain_task(self):
        """Periodic retraining task."""
        self.logger.info("retrain_task_triggered")
        
        # Get available 1m bars - NO MINIMUM REQUIRED
        num_bars = self.buffer_manager.get_bar_count('1m')
        
        if num_bars == 0:
            self.logger.warning("no_data_for_retraining")
            return
        
        # Get training data
        df_1m = self.buffer_manager.get_rolling_df(
            self.config.data.symbol,
            freq='1m',
            window_hours=self.config.data.rolling_window_hours
        )
        
        # Compute features for drift detection
        X, y = self.feature_service.compute_train_features(df_1m)
        
        if X.empty:
            self.logger.warning("no_features_for_retraining")
            return
        
        # Set baseline if first time
        if not self.monitor.baseline_features:
            self.monitor.set_baseline_features(X)
        
        # Check for triggers
        drift_detected, psi_scores = self.monitor.check_drift(X)
        performance_degraded = self.monitor.check_performance_degradation()
        
        # Get current model metadata
        current_metadata = self.model_registry.load_metadata("active")
        current_metrics = current_metadata.get('metrics') if current_metadata else None
        
        # Retrain if triggered
        if drift_detected or performance_degraded or current_metrics is None:
            self.logger.info(
                "retraining_triggered",
                drift=drift_detected,
                performance_degraded=performance_degraded
            )
            
            try:
                # Train new model
                model, metadata = self.trainer.train(df_1m)
                
                # Register as candidate
                version = self.model_registry.register(model, metadata, status="candidate")
                
                # Promote to staging
                self.model_registry.promote(version, target="staging")
                
                # Enter shadow mode
                self.shadow_mode = True
                self.shadow_predictions = []
                
                self.logger.info(
                    "shadow_mode_started",
                    version=version,
                    duration_minutes=self.config.retraining.shadow_mode_minutes
                )
                
                # Schedule shadow mode evaluation
                asyncio.create_task(self._evaluate_shadow_mode(version))
                
            except Exception as e:
                self.logger.error("retraining_failed", error=str(e))
        
        self.last_retrain_time = datetime.now()
    
    async def _evaluate_shadow_mode(self, candidate_version: str):
        """Evaluate shadow mode and promote if successful."""
        shadow_duration = self.config.retraining.shadow_mode_minutes * 60
        await asyncio.sleep(shadow_duration)
        
        min_predictions = self.config.retraining.min_shadow_predictions
        
        if len(self.shadow_predictions) < min_predictions:
            self.logger.warning(
                "insufficient_shadow_predictions",
                got=len(self.shadow_predictions),
                required=min_predictions
            )
            self.shadow_mode = False
            return
        
        # Check shadow performance
        import numpy as np
        shadow_pred_values = [p.predicted_return for p in self.shadow_predictions]
        pred_std = np.std(shadow_pred_values)
        
        # Simple check: predictions should have reasonable variance
        if pred_std > 0.0001:  # Not stuck at constant value
            # Promote to active
            self.model_registry.promote(candidate_version, target="active")
            self.logger.info("candidate_promoted_to_active", version=candidate_version)
            
            # Exit shadow mode
            self.shadow_mode = False
            self.shadow_predictions = []
            
            # Update baseline features
            df_1m = self.buffer_manager.get_rolling_df(
                self.config.data.symbol,
                freq='1m'
            )
            X, _ = self.feature_service.compute_train_features(df_1m)
            self.monitor.set_baseline_features(X)
        else:
            self.logger.warning("shadow_model_rejected", pred_std=pred_std)
            self.shadow_mode = False
    
    async def _metrics_task(self):
        """Periodic metrics logging task."""
        position = self.execution_engine.get_position()
        daily_pnl = self.execution_engine.get_daily_pnl()
        pred_stats = self.predictor.get_prediction_statistics()
        
        metrics = {
            'position_quantity': position.quantity,
            'position_pnl': position.unrealized_pnl + position.realized_pnl,
            'daily_pnl': daily_pnl,
            'prediction_count': pred_stats.get('count', 0),
            'prediction_mean': pred_stats.get('mean', 0),
            'buffer_1m_size': self.buffer_manager.get_bar_count('1m')
        }
        
        self.monitor.log_metrics(metrics)
    
    def _setup_scheduled_tasks(self):
        """Setup periodic tasks with APScheduler."""
        # Retraining every N minutes
        self.scheduler.add_job(
            self._retrain_task,
            trigger=IntervalTrigger(minutes=self.config.training.train_freq_minutes),
            id='retrain_task',
            name='Model Retraining',
            replace_existing=True
        )
        
        # Metrics logging every minute
        self.scheduler.add_job(
            self._metrics_task,
            trigger=IntervalTrigger(seconds=self.config.monitoring.metrics_interval_seconds),
            id='metrics_task',
            name='Metrics Logging',
            replace_existing=True
        )
        
        self.logger.info("scheduled_tasks_configured")
    
    async def _fetch_historical_and_train(self):
        """Fetch historical 1-minute data from Binance and train initial model."""
        self.logger.info("fetching_historical_data_for_initial_training")
        
        from binance.client import Client
        import os
        import pandas as pd
        
        try:
            # Initialize Binance client
            api_key = os.getenv(self.config.binance.api_key_env)
            api_secret = os.getenv(self.config.binance.api_secret_env)
            client = Client(api_key, api_secret)
            
            # Fetch last 24 hours of 1-minute data (should give us 1440 rows)
            hours = self.config.data.rolling_window_hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            self.logger.info(
                "fetching_klines",
                symbol=self.config.data.symbol,
                hours=hours,
                expected_rows=hours * 60
            )
            
            # Fetch with limit to ensure we get full data
            klines = client.get_historical_klines(
                symbol=self.config.data.symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                start_str=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end_str=end_time.strftime('%Y-%m-%d %H:%M:%S'),
                limit=1500  # Request extra to ensure full coverage
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
            df['symbol'] = self.config.data.symbol
            
            # Keep only required columns
            df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'trades']]
            df.rename(columns={'trades': 'num_trades'}, inplace=True)
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Fill missing timestamps (if any gaps in data)
            df = self._fill_missing_timestamps(df)
            
            # Impute missing values (forward fill, then backward fill)
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].ffill().bfill()
            df['volume'] = df['volume'].fillna(0.0)
            df['num_trades'] = df['num_trades'].fillna(0)
            
            self.logger.info(
                "historical_data_fetched",
                num_bars=len(df),
                price_range=f"${df['close'].min():.2f} - ${df['close'].max():.2f}",
                expected_bars=hours * 60,
                coverage_pct=f"{len(df) / (hours * 60) * 100:.1f}%"
            )
            
            # Store in buffer manager
            for _, row in df.iterrows():
                self.buffer_manager.add_bar_1m(row.to_dict())
            
            # Train initial model
            self.logger.info("training_initial_model")
            model, metadata = self.trainer.train(df)
            
            # Register and promote to active
            version = self.model_registry.register(model, metadata, status="staging")
            self.model_registry.promote(version, target="active")
            
            self.logger.info(
                "initial_model_trained",
                version=version,
                metrics=metadata.get('metrics')
            )
            
        except Exception as e:
            self.logger.error("initial_training_failed", error=str(e))
            raise
    
    def _fill_missing_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing minute timestamps with forward-filled data."""
        if df.empty:
            return df
        
        # Create complete minute range
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        complete_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        
        # Create complete dataframe
        complete_df = pd.DataFrame({'timestamp': complete_range})
        
        # Merge with original data
        df = complete_df.merge(df, on='timestamp', how='left')
        
        # Fill symbol
        df['symbol'] = self.config.data.symbol
        
        return df

    async def start(self):
        """Start the trading bot."""
        self.logger.info("starting_trading_bot")
        self.running = True
        
        # Fetch historical data and train initial model
        await self._fetch_historical_and_train()
        
        # Setup scheduled tasks
        self._setup_scheduled_tasks()
        self.scheduler.start()
        
        # Start ingestion
        await self.ingestor.start()
    
    async def stop(self):
        """Stop the trading bot."""
        self.logger.info("stopping_trading_bot")
        self.running = False
        
        # Stop scheduler
        self.scheduler.shutdown()
        
        # Stop ingestor
        await self.ingestor.stop()
        
        # Log final status
        status = self.monitor.get_system_status()
        self.logger.info("final_status", status=status)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("shutdown_signal_received")
        asyncio.create_task(self.stop())
        sys.exit(0)


async def main():
    """Main entry point."""
    bot = TradingBot()
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
