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
            level="INFO"
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
        self.ingestor = BinanceIngestor(self.config, on_tick_callback=self._on_tick)
        
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
    
    async def _on_tick(self, tick):
        """Callback for each incoming tick."""
        # Aggregate tick into bars
        completed_bars = self.aggregator.add_tick(tick)
        
        # Handle completed 1-second bar
        if completed_bars['1s']:
            bar_1s = completed_bars['1s']
            self.buffer_manager.add_bar_1s({
                'timestamp': bar_1s.timestamp,
                'symbol': bar_1s.symbol,
                'open': bar_1s.open,
                'high': bar_1s.high,
                'low': bar_1s.low,
                'close': bar_1s.close,
                'volume': bar_1s.volume,
                'num_trades': bar_1s.num_trades
            })
            
            # Generate prediction every second
            await self._generate_prediction()
        
        # Handle completed 1-minute bar
        if completed_bars['1m']:
            bar_1m = completed_bars['1m']
            self.buffer_manager.add_bar_1m({
                'timestamp': bar_1m.timestamp,
                'symbol': bar_1m.symbol,
                'open': bar_1m.open,
                'high': bar_1m.high,
                'low': bar_1m.low,
                'close': bar_1m.close,
                'volume': bar_1m.volume,
                'num_trades': bar_1m.num_trades
            })
    
    async def _generate_prediction(self):
        """Generate prediction and make trading decision."""
        # Get data
        df_1m = self.buffer_manager.get_rolling_df(
            self.config.data.symbol,
            freq='1m'
        )
        
        df_1s = self.buffer_manager.get_rolling_df(
            self.config.data.symbol,
            freq='1s'
        )
        
        if df_1m.empty:
            return
        
        # Reload model if updated
        self.predictor.reload_model_if_updated()
        
        # Generate prediction
        prediction = self.predictor.predict(df_1m, df_1s.tail(10) if not df_1s.empty else None)
        
        if prediction is None:
            return
        
        # Shadow mode: collect predictions without trading
        if self.shadow_mode:
            self.shadow_predictions.append(prediction)
            self.logger.info("shadow_prediction", prediction=prediction.predicted_return)
            return
        
        # Get current price and volatility
        current_price = df_1m.iloc[-1]['close']
        recent_returns = df_1m['close'].pct_change().dropna()
        volatility = recent_returns.std() if len(recent_returns) > 1 else 0.02
        
        # Check stop-loss and take-profit on existing position
        if self.execution_engine.position.quantity != 0:
            if self.execution_engine.check_stop_loss(current_price):
                # Close position due to stop-loss
                close_side = OrderSide.SELL if self.execution_engine.position.quantity > 0 else OrderSide.BUY
                self.execution_engine.submit_order(
                    side=close_side,
                    quantity=abs(self.execution_engine.position.quantity),
                    order_type=OrderType.MARKET
                )
                self.logger.info("position_closed_stop_loss")
                return
            
            if self.execution_engine.check_take_profit(current_price):
                # Close position due to take-profit
                close_side = OrderSide.SELL if self.execution_engine.position.quantity > 0 else OrderSide.BUY
                self.execution_engine.submit_order(
                    side=close_side,
                    quantity=abs(self.execution_engine.position.quantity),
                    order_type=OrderType.MARKET
                )
                self.logger.info("position_closed_take_profit")
                return
        
        # Make trading decision
        decision = self.execution_engine.make_decision(
            prediction=prediction.predicted_return,
            current_price=current_price,
            volatility=volatility
        )
        
        if decision:
            # Execute trade
            order = self.execution_engine.submit_order(
                side=decision['side'],
                quantity=decision['size'],
                order_type=decision['order_type']
            )
            
            if order:
                # Log trade
                trade_data = {
                    'timestamp': order.timestamp,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'price': order.filled_price,
                    'prediction': prediction.predicted_return,
                    'model_version': prediction.model_version,
                    'pnl': 0.0  # Will be updated on exit
                }
                self.monitor.log_trade(trade_data)
    
    def _import_numpy(self):
        """Import numpy for shadow mode evaluation."""
        import numpy as np
        return np
    
    async def _retrain_task(self):
        """Periodic retraining task."""
        self.logger.info("retrain_task_triggered")
        
        # Check if we have sufficient data
        if not self.buffer_manager.has_sufficient_data(
            min_rows=self.config.training.min_training_rows,
            freq='1m'
        ):
            self.logger.warning("insufficient_data_for_retraining")
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
    
    async def start(self):
        """Start the trading bot."""
        self.logger.info("starting_trading_bot")
        self.running = True
        
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
