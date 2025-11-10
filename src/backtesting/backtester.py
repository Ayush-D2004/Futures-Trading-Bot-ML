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
        self.position = 0.0
        self.cash = config.backtesting.initial_capital
        self.entry_price = 0.0
    
    def run(
        self,
        df_1m: pd.DataFrame,
        df_1s: Optional[pd.DataFrame] = None,
        model_version: str = "active"
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            df_1m: Historical 1-minute bars
            df_1s: Historical 1-second bars (optional)
            model_version: Model version to use
            
        Returns:
            dict: Backtest results with metrics and trades
        """
        self.logger.info("backtest_started", num_bars=len(df_1m))
        
        # Load model
        model = self.model_registry.load_model(model_version)
        if model is None:
            raise ValueError(f"Model {model_version} not found")
        
        # Reset state
        self.trades = []
        self.equity_curve = []
        self.position = 0.0
        self.cash = self.config.backtesting.initial_capital
        
        # Compute features and labels
        X, y = self.feature_service.compute_train_features(df_1m)
        
        if X.empty:
            raise ValueError("No features computed from data")
        
        # Generate predictions
        predictions = model.predict(X)
        
        # Simulate trading loop
        for i in range(len(X)):
            timestamp = df_1m.iloc[i]['timestamp']
            current_price = df_1m.iloc[i]['close']
            prediction = predictions[i]
            
            # Make decision
            decision = self._make_decision(prediction, current_price)
            
            if decision:
                # Execute trade
                self._execute_trade(
                    timestamp=timestamp,
                    side=decision['side'],
                    quantity=decision['quantity'],
                    price=current_price,
                    prediction=prediction
                )
            
            # Update equity
            equity = self._calculate_equity(current_price)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': self.cash,
                'position_value': self.position * current_price
            })
        
        # Calculate metrics
        results = self._calculate_metrics()
        
        self.logger.info("backtest_completed", results=results)
        
        return results
    
    def _make_decision(self, prediction: float, current_price: float) -> Optional[Dict]:
        """Simulate trading decision logic."""
        # Cost estimation
        fee_rate = self.config.execution.fee_rate
        slippage_rate = self.config.execution.slippage_rate
        total_cost = fee_rate + slippage_rate
        
        # Edge threshold
        min_edge = self.config.execution.min_edge_bps / 10000
        
        # No trade if edge too small
        if abs(prediction) < (total_cost + min_edge):
            return None
        
        # Position sizing (simplified)
        position_value = self.config.execution.base_capital * self.config.execution.position_size_fraction
        quantity = position_value / current_price
        
        # Determine side
        if prediction > 0 and self.position <= 0:
            return {'side': 'BUY', 'quantity': quantity}
        elif prediction < 0 and self.position >= 0:
            return {'side': 'SELL', 'quantity': quantity}
        
        return None
    
    def _execute_trade(
        self,
        timestamp: datetime,
        side: str,
        quantity: float,
        price: float,
        prediction: float
    ):
        """Execute simulated trade with costs."""
        # Apply slippage
        if side == 'BUY':
            exec_price = price * (1 + self.config.execution.slippage_rate)
        else:
            exec_price = price * (1 - self.config.execution.slippage_rate)
        
        # Calculate costs
        trade_value = quantity * exec_price
        fee = trade_value * self.config.execution.fee_rate
        
        # Execute
        if side == 'BUY':
            # Close short if exists
            if self.position < 0:
                pnl = (self.entry_price - exec_price) * abs(self.position)
                self.cash += pnl - fee
                
                trade_record = {
                    'timestamp': timestamp,
                    'side': 'CLOSE_SHORT',
                    'quantity': abs(self.position),
                    'price': exec_price,
                    'pnl': pnl - fee,
                    'prediction': prediction
                }
                self.trades.append(trade_record)
                
                self.position = 0
            
            # Open long
            self.position = quantity
            self.entry_price = exec_price
            self.cash -= (trade_value + fee)
            
        else:  # SELL
            # Close long if exists
            if self.position > 0:
                pnl = (exec_price - self.entry_price) * self.position
                self.cash += pnl - fee
                
                trade_record = {
                    'timestamp': timestamp,
                    'side': 'CLOSE_LONG',
                    'quantity': self.position,
                    'price': exec_price,
                    'pnl': pnl - fee,
                    'prediction': prediction
                }
                self.trades.append(trade_record)
                
                self.position = 0
            
            # Open short
            self.position = -quantity
            self.entry_price = exec_price
            self.cash += (trade_value - fee)
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate total equity."""
        position_value = self.position * current_price
        return self.cash + position_value
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        df_trades = pd.DataFrame(self.trades)
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
