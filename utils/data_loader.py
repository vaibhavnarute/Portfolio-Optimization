import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import seaborn as sns

class DataLoader:
    """
    Utility class for loading and processing financial data.
    """
    
    def __init__(self, data_dir="data", use_cache=True, cache_expiry_days=7):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_dir : str, default="data"
            Directory to store cached data
        use_cache : bool, default=True
            Whether to use cached data
        cache_expiry_days : int, default=7
            Number of days after which cached data expires
        """
        self.data_dir = data_dir
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
    def load_yahoo_finance(self, tickers, start_date, end_date=None, interval='1d', use_cache=True):
        """
        Load data from Yahoo Finance.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        interval : str, default='1d'
            Data interval (1d, 1wk, 1mo, etc.)
        use_cache : bool, default=True
            Whether to use cached data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with stock prices
        """
        # Set end date to today if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Create cache directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Create cache filename
        tickers_str = '_'.join(tickers)
        cache_file = f"data/yf_{tickers_str}_{start_date}_{end_date}_{interval}.pkl"
        
        # Check if cache file exists
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            return pd.read_pickle(cache_file)
        
        # Download data
        print(f"Downloading data for {tickers} from {start_date} to {end_date}")
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
        
        # Handle single ticker case properly
        if len(tickers) == 1:
            # If we have a single ticker, the columns won't be a MultiIndex
            # We need to convert it to match the format of multiple tickers
            if not isinstance(data.columns, pd.MultiIndex):
                # Create column names that match the format when multiple tickers are used
                columns = data.columns
                ticker = tickers[0]
                data.columns = pd.MultiIndex.from_product([columns, [ticker]])
        
        # Save to cache
        data.to_pickle(cache_file)
        
        return data
    
    def calculate_returns(self, prices, method="simple"):
        """
        Calculate returns from price data.
        
        Parameters:
        -----------
        prices : pandas.DataFrame
            DataFrame with price data
        method : str, default="simple"
            Method to calculate returns ("simple" or "log")
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with returns
        """
        if method == "simple":
            returns = prices.pct_change().dropna()
        elif method == "log":
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
            
        return returns
    
    def calculate_risk_metrics(self, returns, risk_free_rate=0.0, periods_per_year=252):
        """
        Calculate risk metrics from returns data.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with returns data
        risk_free_rate : float, default=0.0
            Risk-free rate (annualized)
        periods_per_year : int, default=252
            Number of periods per year (252 for daily, 52 for weekly, 12 for monthly)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with risk metrics
        """
        # Calculate annualized return
        ann_return = returns.mean() * periods_per_year
        
        # Calculate annualized volatility
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        
        # Calculate Sharpe ratio
        daily_rf = risk_free_rate / periods_per_year
        sharpe = (returns.mean() - daily_rf) / returns.std() * np.sqrt(periods_per_year)
        
        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Calculate skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Combine metrics into a DataFrame
        metrics = pd.DataFrame({
            'Ann. Return': ann_return,
            'Ann. Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95
        })
        
        return metrics
    
    def calculate_rolling_metrics(self, returns, window=252, risk_free_rate=0.0):
        """
        Calculate rolling risk metrics.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with returns data
        window : int, default=252
            Rolling window size
        risk_free_rate : float, default=0.0
            Risk-free rate (annualized)
            
        Returns:
        --------
        dict
            Dictionary with rolling metrics DataFrames
        """
        # Calculate daily risk-free rate
        daily_rf = risk_free_rate / 252
        
        # Calculate rolling metrics
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (returns.rolling(window).mean() - daily_rf) / returns.rolling(window).std() * np.sqrt(252)
        
        # Calculate rolling drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.rolling(window, min_periods=1).max()
        rolling_drawdown = (cum_returns / rolling_max - 1)
        
        return {
            'return': rolling_return,
            'volatility': rolling_vol,
            'sharpe': rolling_sharpe,
            'drawdown': rolling_drawdown
        }
    
    def calculate_correlation_matrix(self, returns):
        """
        Calculate correlation matrix of returns.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with returns data
            
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix
        """
        return returns.corr()
    
    def plot_prices(self, prices, title="Stock Prices", figsize=(12, 8), normalize=True):
        """
        Plot stock prices.
        
        Parameters:
        -----------
        prices : pandas.DataFrame
            DataFrame with price data
        title : str, default="Stock Prices"
            Plot title
        figsize : tuple, default=(12, 8)
            Figure size
        normalize : bool, default=True
            Whether to normalize prices to start at 100
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if normalize:
            # Normalize prices to start at 100
            normalized = prices / prices.iloc[0] * 100
            normalized.plot(ax=ax)
            ylabel = 'Normalized Price (100 = Start)'
        else:
            # Plot raw prices
            prices.plot(ax=ax)
            ylabel = 'Price'
        
        # Add labels and title
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig
    
    def plot_returns(self, returns, title="Stock Returns", figsize=(12, 8)):
        """
        Plot stock returns.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with returns data
        title : str, default="Stock Returns"
            Plot title
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot returns
        returns.plot(ax=ax)
        
        # Add labels and title
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Returns', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig
    
    def plot_correlation_matrix(self, returns, title="Correlation Matrix", figsize=(10, 8)):
        """
        Plot correlation matrix of returns.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with returns data
        title : str, default="Correlation Matrix"
            Plot title
        figsize : tuple, default=(10, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Calculate correlation matrix
        corr = returns.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   linewidths=0.5, ax=ax, fmt='.2f')
        
        # Add title
        ax.set_title(title, fontsize=14)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def plot_rolling_metrics(self, returns, window=252, risk_free_rate=0.0, figsize=(15, 10)):
        """
        Plot rolling risk metrics.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with returns data
        window : int, default=252
            Rolling window size
        risk_free_rate : float, default=0.0
            Risk-free rate (annualized)
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Calculate rolling metrics
        rolling_metrics = self.calculate_rolling_metrics(returns, window, risk_free_rate)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot rolling return
        rolling_metrics['return'].plot(ax=axes[0])
        axes[0].set_title('Rolling Annualized Return', fontsize=12)
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Return')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Plot rolling volatility
        rolling_metrics['volatility'].plot(ax=axes[1])
        axes[1].set_title('Rolling Annualized Volatility', fontsize=12)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Volatility')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        # Plot rolling Sharpe ratio
        rolling_metrics['sharpe'].plot(ax=axes[2])
        axes[2].set_title('Rolling Sharpe Ratio', fontsize=12)
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Sharpe Ratio')
        axes[2].grid(True, linestyle='--', alpha=0.5)
        
        # Plot rolling drawdown
        rolling_metrics['drawdown'].plot(ax=axes[3])
        axes[3].set_title('Rolling Drawdown', fontsize=12)
        axes[3].set_xlabel('Date')
        axes[3].set_ylabel('Drawdown')
        axes[3].grid(True, linestyle='--', alpha=0.5)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def get_sector_data(self, tickers):
        """
        Get sector information for tickers.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sector information
        """
        sector_data = {}
        
        print("Fetching sector data...")
        for ticker in tqdm(tickers):
            try:
                # Get ticker info
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                # Extract sector and industry
                sector = info.get('sector', 'Unknown')
                industry = info.get('industry', 'Unknown')
                
                sector_data[ticker] = {
                    'Sector': sector,
                    'Industry': industry
                }
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                sector_data[ticker] = {
                    'Sector': 'Unknown',
                    'Industry': 'Unknown'
                }
        
        return pd.DataFrame.from_dict(sector_data, orient='index')