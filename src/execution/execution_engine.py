"""
Execution engine for order placement and risk management.
"""
import os
from datetime import datetime
from typing import Optional, Dict
from binance.client import Client
from binance.exceptions import BinanceAPIException
import uuid

from src.config import Config
from src.schemas import Order, OrderSide, OrderType, OrderStatus, Position
from src.utils import get_logger


class ExecutionEngine:
    """
    Manages order execution, position tracking, and risk controls.
    Integrates with Binance API (testnet for development).
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("execution_engine")
        
        # Load API credentials
        api_key = os.getenv(config.binance.api_key_env)
        api_secret = os.getenv(config.binance.api_secret_env)
        
        if not api_key or not api_secret:
            self.logger.warning(
                "binance_credentials_not_found",
                msg="Set credentials in .env file"
            )
        
        # Initialize Binance client (always mainnet for live data)
        # paper_trading controls whether orders are actually placed
        self.client = Client(api_key=api_key, api_secret=api_secret)
        
        # Position tracking
        self.position = Position(
            symbol=config.data.symbol,
            quantity=0.0,
            avg_entry_price=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            last_update=datetime.now()
        )
        
        # Risk state
        self.daily_pnl = 0.0
        self.daily_start = datetime.now().date()
        self.kill_switch_active = False
        
        # Peak PnL tracking for drawdown limit
        self.peak_total_pnl = 0.0
        
    def make_decision(
        self, 
        prediction: float,
        current_price: float,
        volatility: float
    ) -> Optional[Dict]:
        """
        Make trading decision based on prediction and risk parameters.
        
        Args:
            prediction: Predicted 5-minute return
            current_price: Current market price
            volatility: Realized volatility
            
        Returns:
            Decision dict with side, size, order_type or None
        """
        # Kill switch check
        if self.kill_switch_active:
            self.logger.warning("kill_switch_active")
            return None
        
        # Daily loss limit check
        if self._check_daily_drawdown():
            self.logger.warning("daily_drawdown_limit_reached")
            self.kill_switch_active = True
            return None
        
        # Peak PnL drawdown check (20% from peak)
        if self._check_peak_pnl_drawdown():
            self.logger.error("peak_pnl_drawdown_limit_reached")
            self.kill_switch_active = True
            return None
        
        # Cost estimation
        cost_est = self._estimate_costs(current_price)
        
        # Edge threshold
        min_edge = self.config.execution.min_edge_bps / 10000  # Convert bps to decimal
        
        # Decision logic
        if abs(prediction) < (cost_est + min_edge):
            return None  # No trade
        
        # Determine side
        side = OrderSide.BUY if prediction > 0 else OrderSide.SELL
        
        # Position sizing
        size = self._calculate_position_size(
            current_price=current_price,
            volatility=volatility,
            predicted_return=prediction
        )
        
        if size == 0:
            return None
        
        # Order type
        order_type = OrderType[self.config.execution.order_type]
        
        decision = {
            'side': side,
            'size': size,
            'order_type': order_type,
            'prediction': prediction,
            'edge': abs(prediction) - cost_est
        }
        
        self.logger.info("trading_decision", decision=decision)
        
        return decision
    
    def submit_order(
        self,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.LIMIT,
        price: Optional[float] = None
    ) -> Optional[Order]:
        """
        Submit order to Binance.
        
        Args:
            side: BUY or SELL
            quantity: Order quantity
            order_type: LIMIT or MARKET
            price: Limit price (required for LIMIT orders)
            
        Returns:
            Order object or None
        """
        if self.config.execution.paper_trading:
            return self._simulate_order(side, quantity, order_type, price)
        
        try:
            client_order_id = f"TRC_{uuid.uuid4().hex[:8]}"
            
            if order_type == OrderType.MARKET:
                response = self.client.create_order(
                    symbol=self.config.data.symbol,
                    side=side.value,
                    type='MARKET',
                    quantity=quantity,
                    newClientOrderId=client_order_id
                )
            else:  # LIMIT
                if price is None:
                    raise ValueError("Price required for LIMIT orders")
                
                response = self.client.create_order(
                    symbol=self.config.data.symbol,
                    side=side.value,
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=quantity,
                    price=price,
                    newClientOrderId=client_order_id
                )
            
            order = Order(
                symbol=self.config.data.symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                client_order_id=client_order_id,
                status=OrderStatus[response['status']],
                filled_quantity=float(response.get('executedQty', 0)),
                filled_price=float(response.get('fills', [{}])[0].get('price', 0)) if response.get('fills') else None,
                timestamp=datetime.now()
            )
            
            # Update position
            self._update_position(order)
            
            self.logger.info("order_submitted", order=order)
            
            return order
            
        except BinanceAPIException as e:
            self.logger.error("binance_api_error", error=str(e))
            return None
        except Exception as e:
            self.logger.error("order_submission_error", error=str(e))
            return None
    
    def _simulate_order(
        self,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        price: Optional[float]
    ) -> Order:
        """Simulate order for paper trading."""
        # Get current market price
        ticker = self.client.get_symbol_ticker(symbol=self.config.data.symbol)
        market_price = float(ticker['price'])
        
        # Apply slippage
        if order_type == OrderType.MARKET:
            if side == OrderSide.BUY:
                fill_price = market_price * (1 + self.config.execution.slippage_rate)
            else:
                fill_price = market_price * (1 - self.config.execution.slippage_rate)
        else:
            fill_price = price
        
        order = Order(
            symbol=self.config.data.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            client_order_id=f"PAPER_{uuid.uuid4().hex[:8]}",
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=fill_price,
            timestamp=datetime.now()
        )
        
        # Update position
        self._update_position(order)
        
        self.logger.info("paper_order_executed", order=order)
        
        return order
    
    def _update_position(self, order: Order):
        """Update position state after order fill."""
        if order.status != OrderStatus.FILLED:
            return
        
        quantity_delta = order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity
        
        # Update position
        old_quantity = self.position.quantity
        new_quantity = old_quantity + quantity_delta
        
        if new_quantity == 0:
            # Position closed
            if old_quantity != 0:
                pnl = (order.filled_price - self.position.avg_entry_price) * abs(old_quantity)
                if old_quantity < 0:  # Short position
                    pnl = -pnl
                
                self.position.realized_pnl += pnl
                self.daily_pnl += pnl
                
                self.logger.info("position_closed", pnl=pnl)
            
            self.position.quantity = 0.0
            self.position.avg_entry_price = 0.0
        else:
            # Position opened or increased
            if old_quantity == 0:
                self.position.avg_entry_price = order.filled_price
            else:
                # Weighted average entry
                total_cost = (old_quantity * self.position.avg_entry_price) + (quantity_delta * order.filled_price)
                self.position.avg_entry_price = total_cost / new_quantity
            
            self.position.quantity = new_quantity
        
        self.position.last_update = datetime.now()
    
    def _calculate_position_size(
        self,
        current_price: float,
        volatility: float,
        predicted_return: float
    ) -> float:
        """
        Calculate position size with volatility scaling.
        
        Uses fractional Kelly criterion with volatility adjustment.
        """
        base_capital = self.config.execution.base_capital
        size_fraction = self.config.execution.position_size_fraction
        target_vol = self.config.execution.target_volatility
        
        # Volatility scaling
        if volatility > 0:
            vol_scalar = target_vol / volatility
            vol_scalar = min(vol_scalar, 2.0)  # Cap at 2x
        else:
            vol_scalar = 1.0
        
        # Position value
        position_value = base_capital * size_fraction * vol_scalar
        
        # Cap at max position
        position_value = min(position_value, self.config.execution.max_position_usd)
        
        # Convert to quantity
        quantity = position_value / current_price
        
        # Round to appropriate precision (typically 3-5 decimals for crypto)
        quantity = round(quantity, 5)
        
        return quantity
    
    def _estimate_costs(self, price: float) -> float:
        """Estimate total trading costs (fees + slippage)."""
        fee_rate = self.config.execution.fee_rate
        slippage_rate = self.config.execution.slippage_rate
        
        total_cost = fee_rate + slippage_rate
        
        return total_cost
    
    def _check_daily_drawdown(self) -> bool:
        """Check if daily drawdown limit exceeded."""
        today = datetime.now().date()
        
        # Reset daily PnL on new day
        if today != self.daily_start:
            self.daily_pnl = 0.0
            self.daily_start = today
            self.kill_switch_active = False
        
        # Check drawdown
        drawdown_pct = (self.daily_pnl / self.config.execution.base_capital) * 100
        
        if drawdown_pct < -self.config.execution.max_daily_drawdown_pct:
            self.logger.error(
                "daily_drawdown_exceeded",
                drawdown_pct=drawdown_pct,
                limit=self.config.execution.max_daily_drawdown_pct
            )
            return True
        
        return False
    
    def _check_peak_pnl_drawdown(self) -> bool:
        """Check if drawdown from peak PnL exceeds limit."""
        total_pnl = self.position.realized_pnl + self.position.unrealized_pnl
        
        # Update peak
        if total_pnl > self.peak_total_pnl:
            self.peak_total_pnl = total_pnl
        
        # Check drawdown from peak
        if self.peak_total_pnl > 0:  # Only check if we've been profitable
            drawdown = (self.peak_total_pnl - total_pnl) / self.peak_total_pnl
            limit = self.config.execution.peak_pnl_drawdown_pct / 100.0
            
            if drawdown > limit:
                self.logger.error(
                    "peak_pnl_drawdown_exceeded",
                    peak_pnl=self.peak_total_pnl,
                    current_pnl=total_pnl,
                    drawdown_pct=drawdown * 100,
                    limit_pct=limit * 100
                )
                return True
        
        return False
    
    def check_stop_loss(self, current_price: float) -> bool:
        """Check if position should be closed due to stop-loss."""
        if self.position.quantity == 0:
            return False
        
        # Calculate unrealized PnL percentage
        if self.position.quantity > 0:  # Long
            pnl_pct = (current_price - self.position.avg_entry_price) / self.position.avg_entry_price
        else:  # Short
            pnl_pct = (self.position.avg_entry_price - current_price) / self.position.avg_entry_price
        
        stop_loss_limit = -self.config.execution.stop_loss_pct / 100.0
        
        if pnl_pct <= stop_loss_limit:
            self.logger.warning(
                "stop_loss_triggered",
                pnl_pct=pnl_pct * 100,
                limit_pct=stop_loss_limit * 100,
                entry_price=self.position.avg_entry_price,
                current_price=current_price
            )
            return True
        
        return False
    
    def check_take_profit(self, current_price: float) -> bool:
        """Check if position should be closed due to take-profit."""
        if self.position.quantity == 0:
            return False
        
        # Calculate unrealized PnL percentage
        if self.position.quantity > 0:  # Long
            pnl_pct = (current_price - self.position.avg_entry_price) / self.position.avg_entry_price
        else:  # Short
            pnl_pct = (self.position.avg_entry_price - current_price) / self.position.avg_entry_price
        
        take_profit_target = self.config.execution.take_profit_pct / 100.0
        
        if pnl_pct >= take_profit_target:
            self.logger.info(
                "take_profit_triggered",
                pnl_pct=pnl_pct * 100,
                target_pct=take_profit_target * 100,
                entry_price=self.position.avg_entry_price,
                current_price=current_price
            )
            return True
        
        return False
    
    def get_position(self) -> Position:
        """Get current position state."""
        return self.position
    
    def get_daily_pnl(self) -> float:
        """Get today's PnL."""
        return self.daily_pnl
