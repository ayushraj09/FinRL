"""
Hybrid Portfolio Optimization + Discrete Trading Environment
For two-stage decision making with AI agent validation

This environment combines:
1. Portfolio optimization for strategic allocation
2. Discrete trading for tactical execution
3. Integration with external validation (AI agents, news, etc.)

Author: FinRL Team
Date: 2025-11-29
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import warnings


class HybridPortfolioTradingEnv(gym.Env):
    """
    Hybrid environment that combines portfolio optimization with discrete trading.
    
    Workflow:
    1. Portfolio optimization determines target weights
    2. Discrete trading environment calculates required trades
    3. External validation (AI agents) filters trades
    4. Approved trades are executed via Alpaca
    
    This environment is designed for periodic rebalancing (e.g., every 2 hours)
    with external validation before execution.
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        initial_amount: float,
        transaction_cost_pct: float = 0.001,
        reward_scaling: float = 1.0,
        tech_indicator_list: List[str] = None,
        time_window: int = 120,  # 120 minutes = 2 hours
        rebalance_interval: int = 120,  # Rebalance every 120 minutes
        trading_start_offset: int = 15,  # Start trading at t+15 minutes
        mode: str = "optimization",  # "optimization" or "trading"
        enable_short: bool = False,
        cash_reserve_ratio: float = 0.1,  # Minimum cash to keep
        max_position_size: float = 0.3,  # Max 30% in single stock
        turbulence_threshold: Optional[float] = None,
        print_verbosity: int = 10,
    ):
        """
        Initialize hybrid environment.
        
        Args:
            df: DataFrame with columns [date, tic, close, high, low, volume, ...tech_indicators]
            stock_dim: Number of stocks in universe
            initial_amount: Initial capital
            transaction_cost_pct: Transaction cost percentage
            reward_scaling: Scaling factor for rewards
            tech_indicator_list: List of technical indicators
            time_window: Lookback window in minutes
            rebalance_interval: How often to rebalance (minutes)
            trading_start_offset: Minutes after market open to start
            mode: "optimization" for portfolio weights, "trading" for discrete actions
            enable_short: Allow short selling
            cash_reserve_ratio: Minimum cash to maintain
            max_position_size: Maximum position size per stock
            turbulence_threshold: Risk threshold for liquidation
            print_verbosity: Print frequency
        """
        
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list or []
        self.time_window = time_window
        self.rebalance_interval = rebalance_interval
        self.trading_start_offset = trading_start_offset
        self.mode = mode
        self.enable_short = enable_short
        self.cash_reserve_ratio = cash_reserve_ratio
        self.max_position_size = max_position_size
        self.turbulence_threshold = turbulence_threshold
        self.print_verbosity = print_verbosity
        
        # State components:
        # [cash, prices(n), holdings(n), tech_indicators(n*m), portfolio_weights(n+1)]
        state_dim = 1 + stock_dim + stock_dim + stock_dim * len(self.tech_indicator_list) + (stock_dim + 1)
        
        # Define action space based on mode
        if mode == "optimization":
            # Portfolio weights: [w1, w2, ..., wn, cash_weight]
            # Softmax normalized to sum to 1
            self.action_space = spaces.Box(
                low=0 if not enable_short else -1, 
                high=1, 
                shape=(stock_dim + 1,),
                dtype=np.float32
            )
        else:  # trading mode
            # Discrete buy/sell actions: [-1, 1] per stock
            self.action_space = spaces.Box(
                low=-1, 
                high=1, 
                shape=(stock_dim,),
                dtype=np.float32
            )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Initialize tracking variables
        self.day = 0
        self.minute = 0
        self.data = None
        self.state = None
        self.terminal = False
        
        # Portfolio tracking
        self.cash = initial_amount
        self.holdings = np.zeros(stock_dim)
        self.portfolio_value = initial_amount
        self.current_weights = np.zeros(stock_dim + 1)
        self.current_weights[-1] = 1.0  # Start with 100% cash
        
        # Memory for analysis
        self.asset_memory = [initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = []
        self.weights_memory = [self.current_weights.copy()]
        self.date_memory = []
        self.trade_proposals_memory = []  # For AI agent validation
        
        # Performance metrics
        self.total_trades = 0
        self.total_transaction_costs = 0
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        
        self.day = 0
        self.minute = self.trading_start_offset
        self.terminal = False
        
        # Reset portfolio
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.stock_dim)
        self.portfolio_value = self.initial_amount
        self.current_weights = np.zeros(self.stock_dim + 1)
        self.current_weights[-1] = 1.0
        
        # Reset memory
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = []
        self.weights_memory = [self.current_weights.copy()]
        self.date_memory = []
        self.trade_proposals_memory = []
        self.total_trades = 0
        self.total_transaction_costs = 0
        
        # Get initial state
        self.data = self._get_current_data()
        self.state = self._get_state()
        
        return self.state, {}
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Portfolio weights (optimization mode) or trade signals (trading mode)
            
        Returns:
            state, reward, terminal, truncated, info
        """
        
        # Check if terminal
        self.terminal = self._is_terminal()
        
        if self.terminal:
            return self._handle_terminal()
        
        # Store action
        self.actions_memory.append(actions.copy())
        
        # Process actions based on mode
        if self.mode == "optimization":
            # Portfolio optimization mode
            target_weights = self._process_portfolio_weights(actions)
            trade_proposals = self._weights_to_trades(target_weights)
        else:
            # Discrete trading mode
            trade_proposals = self._process_discrete_actions(actions)
        
        # Store trade proposals for external validation
        self.trade_proposals_memory.append(trade_proposals)
        
        # Calculate reward before execution (for RL training)
        previous_portfolio_value = self.portfolio_value
        
        # Move to next time step
        self._advance_time()
        
        # Update prices
        self.data = self._get_current_data()
        
        # Calculate new portfolio value (mark-to-market)
        self._update_portfolio_value()
        
        # Calculate reward
        portfolio_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        reward = portfolio_return * self.reward_scaling
        
        # Update memory
        self.portfolio_return_memory.append(portfolio_return)
        self.asset_memory.append(self.portfolio_value)
        self.weights_memory.append(self.current_weights.copy())
        
        # Get new state
        self.state = self._get_state()
        
        # Info dict for external use
        info = {
            "trade_proposals": trade_proposals,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "weights": self.current_weights.copy(),
            "ready_for_validation": self._is_rebalance_time(),
        }
        
        return self.state, reward, self.terminal, False, info
    
    def execute_validated_trades(self, validated_trades: Dict[str, int]) -> Dict:
        """
        Execute trades that have been validated by external AI agents.
        
        Args:
            validated_trades: Dict mapping stock ticker to shares to trade
                             Positive = buy, Negative = sell
                             
        Returns:
            Execution summary with costs and updated portfolio
        """
        
        execution_summary = {
            "executed_trades": {},
            "transaction_costs": 0,
            "cash_before": self.cash,
            "cash_after": 0,
            "portfolio_value_before": self.portfolio_value,
            "portfolio_value_after": 0,
        }
        
        prices = self.data["close"].values
        tickers = self.data["tic"].values
        
        for i, ticker in enumerate(tickers):
            if ticker in validated_trades and validated_trades[ticker] != 0:
                shares = validated_trades[ticker]
                price = prices[i]
                
                if shares > 0:  # Buy
                    cost = shares * price * (1 + self.transaction_cost_pct)
                    if cost <= self.cash:
                        self.holdings[i] += shares
                        self.cash -= cost
                        transaction_cost = shares * price * self.transaction_cost_pct
                        self.total_transaction_costs += transaction_cost
                        self.total_trades += 1
                        execution_summary["executed_trades"][ticker] = shares
                        execution_summary["transaction_costs"] += transaction_cost
                    else:
                        warnings.warn(f"Insufficient cash to buy {shares} shares of {ticker}")
                        
                else:  # Sell
                    shares_to_sell = min(abs(shares), self.holdings[i])
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * price * (1 - self.transaction_cost_pct)
                        self.holdings[i] -= shares_to_sell
                        self.cash += proceeds
                        transaction_cost = shares_to_sell * price * self.transaction_cost_pct
                        self.total_transaction_costs += transaction_cost
                        self.total_trades += 1
                        execution_summary["executed_trades"][ticker] = -shares_to_sell
                        execution_summary["transaction_costs"] += transaction_cost
        
        # Update portfolio value and weights
        self._update_portfolio_value()
        self._update_weights()
        
        execution_summary["cash_after"] = self.cash
        execution_summary["portfolio_value_after"] = self.portfolio_value
        
        return execution_summary
    
    def _get_current_data(self) -> pd.DataFrame:
        """Get current market data for all stocks."""
        # Assuming df has a minute-level timestamp
        # Adjust based on your actual data structure
        current_data = self.df[self.df.index == self.minute].copy()
        return current_data
    
    def _get_state(self) -> np.ndarray:
        """
        Construct state vector.
        
        State: [cash, prices, holdings, tech_indicators, current_weights]
        """
        
        prices = self.data["close"].values
        tech_values = []
        
        for tech in self.tech_indicator_list:
            tech_values.extend(self.data[tech].values)
        
        state = np.hstack([
            [self.cash / self.initial_amount],  # Normalized cash
            prices / 100.0,  # Normalized prices
            self.holdings / 1000.0,  # Normalized holdings
            tech_values if tech_values else [0] * (self.stock_dim * len(self.tech_indicator_list)),
            self.current_weights,
        ])
        
        return state.astype(np.float32)
    
    def _process_portfolio_weights(self, actions: np.ndarray) -> np.ndarray:
        """
        Process portfolio weight actions with constraints.
        
        Args:
            actions: Raw actions from agent
            
        Returns:
            Constrained portfolio weights
        """
        
        # Softmax normalization
        weights = self._softmax_normalization(actions)
        
        # Apply position size limits
        for i in range(self.stock_dim):
            if weights[i] > self.max_position_size:
                excess = weights[i] - self.max_position_size
                weights[i] = self.max_position_size
                weights[-1] += excess  # Add excess to cash
        
        # Enforce minimum cash reserve
        if weights[-1] < self.cash_reserve_ratio:
            deficit = self.cash_reserve_ratio - weights[-1]
            # Take proportionally from stock positions
            stock_weights = weights[:-1]
            if stock_weights.sum() > 0:
                reduction = stock_weights * (deficit / stock_weights.sum())
                weights[:-1] -= reduction
                weights[-1] = self.cash_reserve_ratio
        
        # Renormalize
        weights = weights / weights.sum()
        
        return weights
    
    def _weights_to_trades(self, target_weights: np.ndarray) -> Dict[str, int]:
        """
        Convert target portfolio weights to specific trade proposals.
        
        Args:
            target_weights: Target portfolio weights [w1, ..., wn, cash]
            
        Returns:
            Dict mapping ticker to shares to trade
        """
        
        trade_proposals = {}
        prices = self.data["close"].values
        tickers = self.data["tic"].values
        
        total_value = self.portfolio_value
        
        for i in range(self.stock_dim):
            target_value = total_value * target_weights[i]
            current_value = self.holdings[i] * prices[i]
            
            diff_value = target_value - current_value
            shares_to_trade = int(diff_value / prices[i])
            
            if shares_to_trade != 0:
                trade_proposals[tickers[i]] = shares_to_trade
        
        return trade_proposals
    
    def _process_discrete_actions(self, actions: np.ndarray) -> Dict[str, int]:
        """
        Process discrete trading actions.
        
        Args:
            actions: Buy/sell signals [-1, 1] per stock
            
        Returns:
            Dict mapping ticker to shares to trade
        """
        
        trade_proposals = {}
        prices = self.data["close"].values
        tickers = self.data["tic"].values
        
        # Scale actions
        max_trade_value = self.portfolio_value * 0.1  # 10% per trade
        
        for i in range(self.stock_dim):
            if abs(actions[i]) > 0.1:  # Threshold to avoid tiny trades
                if actions[i] > 0:  # Buy signal
                    max_shares = int(max_trade_value / prices[i])
                    shares = int(actions[i] * max_shares)
                    if shares > 0:
                        trade_proposals[tickers[i]] = shares
                else:  # Sell signal
                    max_shares = int(self.holdings[i])
                    shares = int(abs(actions[i]) * max_shares)
                    if shares > 0:
                        trade_proposals[tickers[i]] = -shares
        
        return trade_proposals
    
    def _update_portfolio_value(self):
        """Update portfolio value based on current prices."""
        prices = self.data["close"].values
        asset_value = np.sum(self.holdings * prices)
        self.portfolio_value = self.cash + asset_value
    
    def _update_weights(self):
        """Update current portfolio weights."""
        prices = self.data["close"].values
        
        for i in range(self.stock_dim):
            self.current_weights[i] = (self.holdings[i] * prices[i]) / self.portfolio_value
        
        self.current_weights[-1] = self.cash / self.portfolio_value
    
    def _is_rebalance_time(self) -> bool:
        """Check if it's time to rebalance."""
        minutes_since_start = self.minute - self.trading_start_offset
        return minutes_since_start > 0 and minutes_since_start % self.rebalance_interval == 0
    
    def _advance_time(self):
        """Advance to next time step."""
        self.minute += 1
        # Add logic for day transitions if needed
    
    def _is_terminal(self) -> bool:
        """Check if episode is terminal."""
        return self.minute >= len(self.df.index.unique()) - 1
    
    def _handle_terminal(self) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Handle terminal state and return final metrics."""
        
        final_return = (self.portfolio_value - self.initial_amount) / self.initial_amount
        
        print("=" * 50)
        print(f"Initial Portfolio Value: ${self.initial_amount:,.2f}")
        print(f"Final Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Total Return: {final_return * 100:.2f}%")
        print(f"Total Trades: {self.total_trades}")
        print(f"Total Transaction Costs: ${self.total_transaction_costs:,.2f}")
        
        # Calculate Sharpe ratio
        returns = np.array(self.portfolio_return_memory)
        if returns.std() > 0:
            sharpe = (252 ** 0.5) * returns.mean() / returns.std()
            print(f"Sharpe Ratio: {sharpe:.3f}")
        
        print("=" * 50)
        
        info = {
            "final_portfolio_value": self.portfolio_value,
            "total_return": final_return,
            "total_trades": self.total_trades,
            "total_costs": self.total_transaction_costs,
        }
        
        return self.state, 0, True, False, info
    
    @staticmethod
    def _softmax_normalization(actions: np.ndarray) -> np.ndarray:
        """Softmax normalization to ensure weights sum to 1."""
        exp_actions = np.exp(actions - np.max(actions))  # Numerical stability
        return exp_actions / exp_actions.sum()
    
    def get_latest_trade_proposals(self) -> Dict[str, int]:
        """
        Get the most recent trade proposals for AI agent validation.
        
        Returns:
            Dict mapping ticker to proposed shares to trade
        """
        if self.trade_proposals_memory:
            return self.trade_proposals_memory[-1]
        return {}
    
    def save_asset_memory(self) -> pd.DataFrame:
        """Save portfolio value history."""
        return pd.DataFrame({
            "portfolio_value": self.asset_memory,
            "returns": self.portfolio_return_memory,
        })
    
    def save_weights_memory(self) -> pd.DataFrame:
        """Save portfolio weights history."""
        weights_df = pd.DataFrame(
            self.weights_memory,
            columns=[f"weight_stock_{i}" for i in range(self.stock_dim)] + ["weight_cash"]
        )
        return weights_df
