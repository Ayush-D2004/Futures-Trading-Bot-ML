"""
Event-driven backtester for strategy validation.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from src.config import Config
from src.features import FeatureService
from src.model import ModelRegistry
from src.utils import (
    get_logger, 
    calculate_sharpe_ratio, 
    calculate_max_drawdown,
    ensure_dir
)


class Backtester:
    """
    Replays historical data and simulates trading strategy.
    Models fees, slippage, and latency for realistic performance estimation.
    """
    
    def __init__(
        self,
        config: Config,
        feature_service: FeatureService,
        model_registry: ModelRegistry
    ):
        self.config = config
        self.feature_service = feature_service
        self.model_registry = model_registry
        self.logger = get_logger("backtester")
        
        # Backtest state
        self.trades = []
        self.equity_curve = []
        self.cash = config.backtesting.initial_capital
        
        # Position state: 'FLAT', 'LONG', 'SHORT'
        self.position_state = 'FLAT'
        self.position_size = 0.0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Trade cooldown to prevent rapid whipsaws
        self.last_trade_time = None
        self.trade_cooldown_minutes = 3  # Wait 3 minutes between trades (reduced from 5)
    
    def run(
        self,
        df_1m: pd.DataFrame,
        initial_model_version: str = "active"
    ) -> Dict:
        """
        Run backtest with rolling retraining strategy.
        
        Strategy:
        1. Train on first 12 hours
        2. Test on remaining data
        3. Retrain every 1 hour during test period
        4. Hold positions across retraining
        
        Args:
            df_1m: Historical 1-minute bars (full dataset)
            initial_model_version: Initial model version to use
            
        Returns:
            dict: Backtest results with metrics and trades
        """
        self.logger.info("backtest_started", num_bars=len(df_1m))
        
        # Reset state
        self.trades = []
        self.equity_curve = []
        self.cash = self.config.backtesting.initial_capital
        self.position_state = 'FLAT'
        self.position_size = 0.0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Split data: train on first 12 hours, test on rest
        train_hours = self.config.backtesting.train_hours
        train_rows = train_hours * 60  # 12 hours = 720 minutes
        
        if len(df_1m) < train_rows:
            raise ValueError(f"Insufficient data: need {train_rows} rows, got {len(df_1m)}")
        
        # Initial training
        train_df = df_1m.iloc[:train_rows].copy()
        test_df = df_1m.iloc[train_rows:].copy()
        
        self.logger.info("initial_training", train_rows=len(train_df), test_rows=len(test_df))
        
        # Train initial model
        from src.model import Trainer
        trainer = Trainer(self.config, self.feature_service)
        model, metadata = trainer.train(train_df)
        
        self.logger.info("initial_model_trained", 
                        val_rmse=metadata['metrics']['val_rmse'],
                        val_directional_accuracy=metadata['metrics']['val_directional_accuracy'])
        
        # Simulate trading on test data with rolling retraining
        retrain_interval = self.config.backtesting.retrain_interval_minutes
        last_retrain_idx = 0
        
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            timestamp = row['timestamp']
            current_price = row['close']
            
            # Retrain every hour
            if i > 0 and i % retrain_interval == 0:
                # Get rolling 12-hour window for retraining
                retrain_start = train_rows + i - (train_hours * 60)
                retrain_end = train_rows + i
                retrain_df = df_1m.iloc[retrain_start:retrain_end].copy()
                
                self.logger.info("retraining", 
                               iteration=i // retrain_interval,
                               rows=len(retrain_df))
                
                model, metadata = trainer.train(retrain_df)
                last_retrain_idx = i
            
            # Get historical context for features (last 12 hours)
            context_start = max(0, train_rows + i - (train_hours * 60))
            context_end = train_rows + i + 1
            context_df = df_1m.iloc[context_start:context_end].copy()
            
            # Generate prediction
            X_live = self.feature_service.compute_live_features(context_df)
            
            if X_live.empty:
                continue
            
            prediction = model.predict(X_live)[0] / 100.0  # Convert from percentage
            
            # Check risk controls on existing position
            if self.position_state != 'FLAT':
                if self._check_stop_loss(current_price):
                    self._close_position(timestamp, current_price, "stop_loss")
                elif self._check_take_profit(current_price):
                    self._close_position(timestamp, current_price, "take_profit")
                elif self._should_reverse(prediction):
                    self._close_position(timestamp, current_price, "signal_reversal")
            
            # Decide action (only if no cooldown active)
            if self._can_trade(timestamp):
                action = self._decide_action(prediction, current_price)
                
                if action in ['OPEN_LONG', 'OPEN_SHORT']:
                    self._open_position(timestamp, current_price, action, prediction)
            
            # Update equity curve
            equity = self._calculate_equity(current_price)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': self.cash,
                'position_value': self.position_size * current_price if self.position_state != 'FLAT' else 0,
                'position_state': self.position_state,
                'unrealized_pnl': self.unrealized_pnl
            })
        
        # DO NOT close remaining position - exclude it from metrics
        # Only count completed round-trip trades
        if self.position_state != 'FLAT':
            final_price = test_df.iloc[-1]['close']
            
            # Calculate what the unrealized PnL would be (for info only, NOT counted)
            if self.position_state == 'LONG':
                unrealized_pnl = (final_price - self.entry_price) * self.position_size
            else:
                unrealized_pnl = (self.entry_price - final_price) * self.position_size
            
            self.logger.warning("open_position_excluded",
                              side=self.position_state,
                              entry_price=self.entry_price,
                              current_price=final_price,
                              size=self.position_size,
                              unrealized_pnl=unrealized_pnl,
                              msg="Position still open at backtest end - EXCLUDED from all metrics")
        else:
            self.logger.info("backtest_no_open_position",
                           msg="No open position at end - all trades are complete")
        
        # Calculate metrics (only uses self.trades, which only contains CLOSED positions)
        results = self._calculate_metrics()
        
        # Log summary
        num_closed = len(self.trades)
        num_excluded = 1 if self.position_state != 'FLAT' else 0
        
        self.logger.info("backtest_completed", 
                        closed_trades=num_closed,
                        excluded_trades=num_excluded,
                        results=results)
        
        return results
    
    def _decide_action(self, prediction: float, current_price: float) -> str:
        """Decide trading action based on prediction."""
        # Cost estimation
        fee_rate = self.config.execution.fee_rate
        slippage_rate = self.config.execution.slippage_rate
        min_edge = self.config.execution.min_edge_bps / 10000
        required_edge = fee_rate + slippage_rate + min_edge
        
        # If already in position, hold it
        if self.position_state != 'FLAT':
            return 'HOLD'
        
        # Check if prediction strong enough
        if abs(prediction) < required_edge:
            return 'NONE'
        
        # Open new position
        if prediction > required_edge:
            return 'OPEN_LONG'
        elif prediction < -required_edge:
            return 'OPEN_SHORT'
        
        return 'NONE'
    
    def _can_trade(self, current_time: datetime) -> bool:
        """Check if enough time has passed since last trade (cooldown)."""
        if self.last_trade_time is None:
            return True
        
        time_since_last_trade = (current_time - self.last_trade_time).total_seconds() / 60.0
        return time_since_last_trade >= self.trade_cooldown_minutes
    
    def _open_position(self, timestamp: datetime, price: float, side: str, prediction: float):
        """Open a new position."""
        # Use backtest-specific position size if available, otherwise use execution config
        position_fraction = getattr(self.config.backtesting, 'position_size_fraction', 
                                   self.config.execution.position_size_fraction)
        position_value = self.config.execution.base_capital * position_fraction
        quantity = position_value / price
        
        # Apply costs
        trade_value = quantity * price
        costs = trade_value * (self.config.execution.fee_rate + self.config.execution.slippage_rate)
        
        self.position_state = 'LONG' if side == 'OPEN_LONG' else 'SHORT'
        self.position_size = quantity
        self.entry_price = price
        
        self.logger.debug("position_opened",
                         timestamp=timestamp,
                         side=self.position_state,
                         price=price,
                         size=quantity,
                         prediction=prediction)
    
    def _close_position(self, timestamp: datetime, price: float, reason: str):
        """Close current position."""
        if self.position_state == 'FLAT':
            return
        
        # Calculate PnL
        if self.position_state == 'LONG':
            pnl = (price - self.entry_price) * self.position_size
        else:  # SHORT
            pnl = (self.entry_price - price) * self.position_size
        
        # Apply costs
        trade_value = self.position_size * price
        costs = trade_value * (self.config.execution.fee_rate + self.config.execution.slippage_rate)
        realized_pnl = pnl - costs
        
        self.realized_pnl += realized_pnl
        self.cash += realized_pnl
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'side': f'CLOSE_{self.position_state}',
            'entry_price': self.entry_price,
            'exit_price': price,
            'quantity': self.position_size,
            'pnl': realized_pnl,
            'reason': reason
        }
        self.trades.append(trade_record)
        
        self.logger.debug("position_closed",
                         timestamp=timestamp,
                         side=self.position_state,
                         pnl=realized_pnl,
                         reason=reason)
        
        # Reset and set cooldown
        self.position_state = 'FLAT'
        self.position_size = 0.0
        self.entry_price = 0.0
        self.last_trade_time = timestamp  # Start cooldown after closing
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate total equity including unrealized PnL."""
        # Calculate unrealized PnL for open position
        if self.position_state != 'FLAT':
            if self.position_state == 'LONG':
                self.unrealized_pnl = (current_price - self.entry_price) * self.position_size
            else:  # SHORT
                self.unrealized_pnl = (self.entry_price - current_price) * self.position_size
        else:
            self.unrealized_pnl = 0.0
        
        return self.cash + self.unrealized_pnl
    
    def _check_stop_loss(self, current_price: float) -> bool:
        """Check if stop-loss triggered."""
        if self.position_state == 'FLAT':
            return False
        
        # Calculate PnL percentage (positive = profit, negative = loss)
        if self.position_state == 'LONG':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # Trigger stop-loss if loss exceeds threshold
        stop_loss_threshold = -self.config.execution.stop_loss_pct / 100.0
        
        if pnl_pct <= stop_loss_threshold:
            self.logger.info("stop_loss_triggered",
                           position=self.position_state,
                           entry_price=self.entry_price,
                           current_price=current_price,
                           pnl_pct=pnl_pct * 100,
                           threshold_pct=stop_loss_threshold * 100)
            return True
        
        return False
    
    def _check_take_profit(self, current_price: float) -> bool:
        """Check if take-profit triggered."""
        if self.position_state == 'FLAT':
            return False
        
        # Calculate PnL percentage (positive = profit, negative = loss)
        if self.position_state == 'LONG':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # Trigger take-profit if profit exceeds threshold
        take_profit_threshold = self.config.execution.take_profit_pct / 100.0
        
        if pnl_pct >= take_profit_threshold:
            self.logger.info("take_profit_triggered",
                           position=self.position_state,
                           entry_price=self.entry_price,
                           current_price=current_price,
                           pnl_pct=pnl_pct * 100,
                           threshold_pct=take_profit_threshold * 100)
            return True
        
        return False
    
    def _should_reverse(self, prediction: float) -> bool:
        """
        Check if signal reversed strongly enough to close position.
        Requires prediction to flip AND exceed threshold to avoid whipsaws.
        """
        # Use half the entry threshold for reversal (more lenient exit)
        fee_rate = self.config.execution.fee_rate
        slippage_rate = self.config.execution.slippage_rate
        min_edge = self.config.execution.min_edge_bps / 10000
        reversal_threshold = (fee_rate + slippage_rate + min_edge) * 0.5  # 50% of entry threshold
        
        # For LONG: reverse if prediction strongly negative
        if self.position_state == 'LONG' and prediction < -reversal_threshold:
            self.logger.info("signal_reversal_detected",
                           position="LONG",
                           prediction=prediction * 100,
                           threshold=-reversal_threshold * 100)
            return True
        
        # For SHORT: reverse if prediction strongly positive
        if self.position_state == 'SHORT' and prediction > reversal_threshold:
            self.logger.info("signal_reversal_detected",
                           position="SHORT",
                           prediction=prediction * 100,
                           threshold=reversal_threshold * 100)
            return True
        
        return False
    
    def _calculate_metrics(self) -> Dict:
        """
        Calculate backtest performance metrics.
        ONLY uses completed (closed) trades from self.trades.
        Any open position at backtest end is automatically excluded.
        """
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Verify we're only counting closed trades
        df_trades = pd.DataFrame(self.trades)
        
        # Extra safety: ensure all trades have both entry and exit
        if not all(['entry_price' in t and 'exit_price' in t for t in self.trades]):
            self.logger.error("incomplete_trades_detected",
                            msg="Some trades missing entry or exit - this should not happen")
        
        df_equity = pd.DataFrame(self.equity_curve)
        
        # PnL metrics
        total_pnl = df_trades['pnl'].sum()
        num_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if (num_trades - winning_trades) > 0 else 0
        
        # Returns
        initial_capital = self.config.backtesting.initial_capital
        final_equity = df_equity.iloc[-1]['equity']
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Equity curve metrics
        equity_series = df_equity['equity']
        returns = equity_series.pct_change().dropna()
        
        sharpe = calculate_sharpe_ratio(returns, periods_per_year=252*24*60)  # Minute bars
        max_dd = calculate_max_drawdown(equity_series)
        
        # Trade statistics
        df_trades['duration'] = df_trades['timestamp'].diff().dt.total_seconds() / 60  # minutes
        avg_duration = df_trades['duration'].mean()
        
        metrics = {
            'total_pnl': float(total_pnl),
            'total_return': float(total_return),
            'num_trades': int(num_trades),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'avg_trade_duration_min': float(avg_duration),
            'initial_capital': float(initial_capital),
            'final_equity': float(final_equity)
        }
        
        return metrics
    
    def save_results(self, results: Dict, trades: List[Dict] = None):
        """Save backtest results to disk."""
        results_dir = Path(self.config.backtesting.results_dir)
        ensure_dir(results_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = results_dir / f"metrics_{timestamp}.json"
        import json
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trades
        if trades or self.trades:
            trades_data = trades if trades else self.trades
            df_trades = pd.DataFrame(trades_data)
            trades_file = results_dir / f"trades_{timestamp}.parquet"
            df_trades.to_parquet(trades_file, index=False)
        
        # Save equity curve
        if self.equity_curve:
            df_equity = pd.DataFrame(self.equity_curve)
            equity_file = results_dir / f"equity_{timestamp}.parquet"
            df_equity.to_parquet(equity_file, index=False)
        
        self.logger.info(
            "backtest_results_saved",
            metrics_file=str(metrics_file)
        )
    
    def plot_results(self):
        """Plot equity curve and drawdown (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            df_equity = pd.DataFrame(self.equity_curve)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curve
            ax1.plot(df_equity['timestamp'], df_equity['equity'])
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True)
            
            # Drawdown
            equity_series = df_equity['equity']
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            
            ax2.fill_between(df_equity['timestamp'], drawdown, 0, alpha=0.3, color='red')
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Time')
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            results_dir = Path(self.config.backtesting.results_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = results_dir / f"equity_plot_{timestamp}.png"
            plt.savefig(plot_file)
            
            self.logger.info("plot_saved", file=str(plot_file))
            
        except ImportError:
            self.logger.warning("matplotlib_not_available")
