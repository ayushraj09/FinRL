"""
Classical Algorithms for Optimal Dynamic Portfolio Management
Real implementations ready to use in your backtest
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


class ExponentialGradientPortfolio:
    """
    Exponential Gradient (EG) Algorithm - RECOMMENDED
    Optimal for all market conditions
    Reference: Helmbold et al. (1998)
    """
    
    def __init__(self, num_stocks: int, learning_rate: float = 0.05):
        self.num_stocks = num_stocks
        self.eta = learning_rate
        self.weights = np.ones(num_stocks) / num_stocks
        self.portfolio_values = [1000000]
        
    def rebalance(self, price_changes: np.ndarray) -> np.ndarray:
        """Update weights based on price changes"""
        log_returns = np.log(price_changes)
        self.weights = self.weights * np.exp(self.eta * log_returns)
        self.weights = self.weights / np.sum(self.weights)
        return self.weights.copy()
    
    def step(self, price_changes: np.ndarray, current_value: float) -> float:
        """Execute one step of portfolio management"""
        weights = self.rebalance(price_changes)
        portfolio_return = np.dot(weights, price_changes)
        new_value = current_value * portfolio_return
        self.portfolio_values.append(new_value)
        return new_value
    
    def get_results(self) -> List[float]:
        return self.portfolio_values


class UniversalPortfolio:
    """
    Follow the Winner (FTW) - Universal Portfolio Algorithm
    Best for trending markets
    Reference: Cover (1991)
    """
    
    def __init__(self, num_stocks: int):
        self.num_stocks = num_stocks
        self.cumulative_returns = np.ones(num_stocks)
        self.portfolio_values = [1000000]
        
    def rebalance(self, current_prices: np.ndarray, 
                  initial_prices: np.ndarray) -> np.ndarray:
        """Weights proportional to cumulative returns"""
        self.cumulative_returns = current_prices / initial_prices
        weights = self.cumulative_returns / np.sum(self.cumulative_returns)
        return weights.copy()
    
    def step(self, current_prices: np.ndarray, 
             initial_prices: np.ndarray,
             previous_prices: np.ndarray,
             current_value: float) -> float:
        """Execute one step"""
        weights = self.rebalance(current_prices, initial_prices)
        price_changes = current_prices / (previous_prices + 1e-10)
        portfolio_return = np.dot(weights, price_changes)
        new_value = current_value * portfolio_return
        self.portfolio_values.append(new_value)
        return new_value
    
    def get_results(self) -> List[float]:
        return self.portfolio_values


class MeanReversionPortfolio:
    """
    Follow the Loser - Mean Reversion Strategy
    Best for oscillating markets
    """
    
    def __init__(self, num_stocks: int, lookback_window: int = 20):
        self.num_stocks = num_stocks
        self.lookback = lookback_window
        self.portfolio_values = [1000000]
        
    def rebalance(self, price_history: pd.DataFrame, 
                  current_time_idx: int) -> np.ndarray:
        """Inverse weights to recent returns"""
        if current_time_idx < self.lookback:
            return np.ones(self.num_stocks) / self.num_stocks
        
        start_idx = current_time_idx - self.lookback
        recent_start = price_history.iloc[start_idx].values
        recent_end = price_history.iloc[current_time_idx].values
        
        returns = recent_end / (recent_start + 1e-10)
        inverse_returns = 1.0 / (returns + 1e-10)
        weights = inverse_returns / np.sum(inverse_returns)
        return weights.copy()
    
    def step(self, price_history: pd.DataFrame,
             current_time_idx: int,
             current_value: float) -> float:
        """Execute one step"""
        if current_time_idx == 0:
            weights = np.ones(self.num_stocks) / self.num_stocks
        else:
            weights = self.rebalance(price_history, current_time_idx)
        
        current_prices = price_history.iloc[current_time_idx].values
        if current_time_idx > 0:
            previous_prices = price_history.iloc[current_time_idx - 1].values
        else:
            previous_prices = current_prices
        
        price_changes = current_prices / (previous_prices + 1e-10)
        portfolio_return = np.dot(weights, price_changes)
        new_value = current_value * portfolio_return
        self.portfolio_values.append(new_value)
        return new_value
    
    def get_results(self) -> List[float]:
        return self.portfolio_values


class DynamicMVOPortfolio:
    """
    Dynamic Mean Variance Optimization
    Recalculates MVO weights every period using rolling window
    """
    
    def __init__(self, num_stocks: int, rolling_window: int = 60):
        self.num_stocks = num_stocks
        self.rolling_window = rolling_window
        self.portfolio_values = [1000000]
        
        try:
            from pypfopt.efficient_frontier import EfficientFrontier
            self.EF = EfficientFrontier
        except ImportError:
            self.EF = None
            print("Warning: pypfopt not installed. Install with: pip install pypfopt")
    
    def calculate_mvo_weights(self, price_history: pd.DataFrame,
                             current_time_idx: int) -> np.ndarray:
        """Calculate MVO optimal weights for current period"""
        if current_time_idx < self.rolling_window:
            return np.ones(self.num_stocks) / self.num_stocks
        
        if self.EF is None:
            return np.ones(self.num_stocks) / self.num_stocks
        
        start_idx = current_time_idx - self.rolling_window
        end_idx = current_time_idx + 1
        window_prices = price_history.iloc[start_idx:end_idx].values
        
        returns = np.diff(window_prices, axis=0) / (window_prices[:-1] + 1e-10)
        mean_returns = np.mean(returns, axis=0)
        cov_returns = np.cov(returns, rowvar=False)
        cov_returns += np.eye(self.num_stocks) * 1e-5
        
        try:
            ef = self.EF(mean_returns, cov_returns, weight_bounds=(0, 0.5))
            ef.max_sharpe()
            weights = ef.get_weights()
            return np.array(weights)
        except:
            return np.ones(self.num_stocks) / self.num_stocks
    
    def step(self, price_history: pd.DataFrame,
             current_time_idx: int,
             current_value: float) -> float:
        """Execute one step"""
        weights = self.calculate_mvo_weights(price_history, current_time_idx)
        
        current_prices = price_history.iloc[current_time_idx].values
        if current_time_idx > 0:
            previous_prices = price_history.iloc[current_time_idx - 1].values
        else:
            previous_prices = current_prices
        
        price_changes = current_prices / (previous_prices + 1e-10)
        portfolio_return = np.dot(weights, price_changes)
        new_value = current_value * portfolio_return
        self.portfolio_values.append(new_value)
        return new_value
    
    def get_results(self) -> List[float]:
        return self.portfolio_values
