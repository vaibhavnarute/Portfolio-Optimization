import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class MonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio performance.
    """
    
    def __init__(self, returns_data, weights, initial_investment=10000):
        """
        Initialize the Monte Carlo simulator.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame
            DataFrame with asset returns
        weights : dict
            Dictionary with asset weights
        initial_investment : float, default=10000
            Initial investment amount
        """
        self.returns_data = returns_data
        self.weights = weights
        self.initial_investment = initial_investment
        self.simulation_results = None
        
    def run_simulation(self, num_simulations=1000, num_periods=252):
        """
        Run Monte Carlo simulation.
        
        Parameters:
        -----------
        num_simulations : int, default=1000
            Number of simulations to run
        num_periods : int, default=252
            Number of periods to simulate (252 trading days = 1 year)
            
        Returns:
        --------
        numpy.ndarray
            Array with simulation results
        """
        # Convert weights dictionary to array
        assets = list(self.weights.keys())
        weights_array = np.array([self.weights[asset] for asset in assets])
        
        # Get mean returns and covariance matrix
        mean_returns = self.returns_data[assets].mean().values
        cov_matrix = self.returns_data[assets].cov().values
        
        # Initialize array to store simulation results
        simulation_results = np.zeros((num_periods + 1, num_simulations))
        simulation_results[0] = self.initial_investment
        
        # Run simulations
        for sim in tqdm(range(num_simulations), desc="Running simulations"):
            # Generate random returns
            random_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, num_periods
            )
            
            # Calculate portfolio returns
            portfolio_returns = np.sum(random_returns * weights_array, axis=1)
            
            # Calculate cumulative portfolio value
            for period in range(num_periods):
                simulation_results[period + 1, sim] = simulation_results[period, sim] * (1 + portfolio_returns[period])
        
        self.simulation_results = simulation_results
        return simulation_results
    
    def calculate_statistics(self):
        """
        Calculate statistics from simulation results.
        
        Returns:
        --------
        dict
            Dictionary with statistics
        """
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        
        # Get final values
        final_values = self.simulation_results[-1]
        
        # Calculate statistics
        mean_final_value = np.mean(final_values)
        median_final_value = np.median(final_values)
        min_final_value = np.min(final_values)
        max_final_value = np.max(final_values)
        percentile_5 = np.percentile(final_values, 5)
        percentile_95 = np.percentile(final_values, 95)
        prob_profit = np.mean(final_values > self.initial_investment)
        
        return {
            'mean_final_value': mean_final_value,
            'median_final_value': median_final_value,
            'min_final_value': min_final_value,
            'max_final_value': max_final_value,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'prob_profit': prob_profit
        }
    
    def plot_simulation_results(self, figsize=(12, 8)):
        """
        Plot simulation results.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot simulations
        for sim in range(self.simulation_results.shape[1]):
            ax.plot(self.simulation_results[:, sim], 'b-', alpha=0.05)
        
        # Plot statistics
        ax.plot(np.median(self.simulation_results, axis=1), 'r-', linewidth=2, label='Median')
        ax.plot(np.percentile(self.simulation_results, 5, axis=1), 'g--', linewidth=2, label='5th Percentile')
        ax.plot(np.percentile(self.simulation_results, 95, axis=1), 'g--', linewidth=2, label='95th Percentile')
        
        # Add labels and title
        ax.set_title('Monte Carlo Simulation', fontsize=14)
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig
    
    def plot_final_value_histogram(self, figsize=(12, 8)):
        """
        Plot histogram of final values.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        
        # Get final values
        final_values = self.simulation_results[-1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.hist(final_values, bins=50, alpha=0.7, color='blue')
        
        # Add vertical lines for statistics
        ax.axvline(np.median(final_values), color='r', linestyle='-', linewidth=2, label='Median')
        ax.axvline(np.percentile(final_values, 5), color='g', linestyle='--', linewidth=2, label='5th Percentile')
        ax.axvline(np.percentile(final_values, 95), color='g', linestyle='--', linewidth=2, label='95th Percentile')
        ax.axvline(self.initial_investment, color='k', linestyle='-', linewidth=2, label='Initial Investment')
        
        # Add labels and title
        ax.set_title('Distribution of Final Portfolio Values', fontsize=14)
        ax.set_xlabel('Portfolio Value ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig