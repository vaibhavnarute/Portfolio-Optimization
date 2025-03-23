import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

class MarkowitzOptimizer:
    """
    Portfolio optimization using Markowitz Modern Portfolio Theory.
    """
    
    def __init__(self, returns_data=None, prices_data=None):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame, optional
            DataFrame with asset returns
        prices_data : pandas.DataFrame, optional
            DataFrame with asset prices
        """
        self.returns_data = returns_data
        self.prices_data = prices_data
        self.ef = None
        
    def optimize_max_sharpe(self, risk_free_rate=0.02):
        """
        Optimize portfolio for maximum Sharpe ratio.
        
        Parameters:
        -----------
        risk_free_rate : float, default=0.02
            Risk-free rate (annualized)
            
        Returns:
        --------
        dict
            Dictionary with optimal weights
        """
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(self.prices_data)
        S = risk_models.sample_cov(self.prices_data)
        
        # Optimize for maximum Sharpe ratio
        self.ef = EfficientFrontier(mu, S)
        weights = self.ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = self.ef.clean_weights()
        
        return cleaned_weights
    
    def optimize_min_volatility(self):
        """
        Optimize portfolio for minimum volatility.
        
        Returns:
        --------
        dict
            Dictionary with optimal weights
        """
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(self.prices_data)
        S = risk_models.sample_cov(self.prices_data)
        
        # Optimize for minimum volatility
        self.ef = EfficientFrontier(mu, S)
        weights = self.ef.min_volatility()
        cleaned_weights = self.ef.clean_weights()
        
        return cleaned_weights
    
    def optimize_efficient_return(self, target_return, risk_free_rate=0.02):
        """
        Optimize portfolio for a target return.
        
        Parameters:
        -----------
        target_return : float
            Target return (annualized)
        risk_free_rate : float, default=0.02
            Risk-free rate (annualized)
            
        Returns:
        --------
        dict
            Dictionary with optimal weights
        """
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(self.prices_data)
        S = risk_models.sample_cov(self.prices_data)
        
        # Optimize for target return
        self.ef = EfficientFrontier(mu, S)
        weights = self.ef.efficient_return(target_return=target_return)
        cleaned_weights = self.ef.clean_weights()
        
        return cleaned_weights
    
    def plot_efficient_frontier(self, risk_free_rate=0.02, points=100, figsize=(10, 7)):
        """
        Plot the efficient frontier.
        
        Parameters:
        -----------
        risk_free_rate : float, default=0.02
            Risk-free rate (annualized)
        points : int, default=100
            Number of points to plot
        figsize : tuple, default=(10, 7)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(self.prices_data)
        S = risk_models.sample_cov(self.prices_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot efficient frontier
        ef = EfficientFrontier(mu, S)
        ef_max_sharpe = EfficientFrontier(mu, S)
        ef_min_vol = EfficientFrontier(mu, S)
        
        # Find optimal portfolios
        weights_max_sharpe = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
        weights_min_vol = ef_min_vol.min_volatility()
        
        # Get performance metrics
        ret_max_sharpe, vol_max_sharpe, _ = ef_max_sharpe.portfolio_performance(risk_free_rate=risk_free_rate)
        ret_min_vol, vol_min_vol, _ = ef_min_vol.portfolio_performance(risk_free_rate=risk_free_rate)
        
        # Generate random portfolios
        n_samples = 10000
        w = np.random.dirichlet(np.ones(len(mu)), n_samples)
        rets = w.dot(mu)
        vols = np.sqrt(np.diag(w @ S @ w.T))
        sharpes = (rets - risk_free_rate) / vols
        
        # Plot random portfolios
        ax.scatter(vols, rets, c=sharpes, cmap='viridis', marker='o', s=10, alpha=0.3)
        
        # Plot efficient frontier
        ef_performance = []
        for target_return in np.linspace(ret_min_vol, max(rets), points):
            ef.efficient_return(target_return=target_return)
            ret, vol, _ = ef.portfolio_performance()
            ef_performance.append((vol, ret))
        
        ef_performance = np.array(ef_performance)
        ax.plot(ef_performance[:, 0], ef_performance[:, 1], 'b--', linewidth=2)
        
        # Plot optimal portfolios
        ax.scatter(vol_max_sharpe, ret_max_sharpe, marker='*', s=200, c='r', label='Maximum Sharpe')
        ax.scatter(vol_min_vol, ret_min_vol, marker='*', s=200, c='g', label='Minimum Volatility')
        
        # Plot capital market line
        ax.plot([0, vol_max_sharpe * 1.5], [risk_free_rate, ret_max_sharpe + (ret_max_sharpe - risk_free_rate) / vol_max_sharpe * vol_max_sharpe * 0.5], 'r-', label='Capital Market Line')
        
        # Add labels and title
        ax.set_title('Efficient Frontier', fontsize=14)
        ax.set_xlabel('Volatility (Standard Deviation)', fontsize=12)
        ax.set_ylabel('Expected Return', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig
    
    def get_discrete_allocation(self, weights, total_portfolio_value=10000):
        """
        Get discrete allocation of assets.
        
        Parameters:
        -----------
        weights : dict
            Dictionary with asset weights
        total_portfolio_value : float, default=10000
            Total portfolio value
            
        Returns:
        --------
        dict
            Dictionary with discrete allocation
        float
            Leftover cash
        """
        # Get latest prices
        latest_prices = get_latest_prices(self.prices_data)
        
        # Create discrete allocation object
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
        
        # Get allocation
        allocation, leftover = da.greedy_portfolio()
        
        return allocation, leftover