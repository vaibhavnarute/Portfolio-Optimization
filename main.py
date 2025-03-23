import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import logging
from pathlib import Path

# Import our models and utilities
from utils.data_loader import DataLoader
from models.markowitz import MarkowitzOptimizer
from models.monte_carlo import MonteCarloSimulator
from models.factor_models import FactorModel
from models.reinforcement import RLPortfolioOptimizer
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("portfolio_optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def download_data(tickers, start_date, end_date):
    """
    Download stock data.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with stock prices
    """
    print(f"Downloading data for {tickers} from {start_date} to {end_date}")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    
    # If we have a single ticker, ensure we have a proper DataFrame
    if len(tickers) == 1:
        data = data.to_frame() if isinstance(data, pd.Series) else data
        
    return data

def run_markowitz_optimization(tickers, start_date, end_date, risk_free_rate=0.02, 
                              transaction_cost=0.001, save_results=True, plot=True):
    """
    Run Markowitz portfolio optimization.
    """
    logger.info(f"Running Markowitz optimization for {tickers}...")
    
    # Download data
    prices_data = download_data(tickers, start_date, end_date)
    
    # Check if 'Adj Close' is available, otherwise use 'Close'
    if 'Adj Close' in prices_data.columns:
        prices = prices_data['Adj Close']
    else:
        # If we have a MultiIndex with 'Close' at level 1
        if isinstance(prices_data.columns, pd.MultiIndex):
            prices = prices_data['Close']
        # If we just have regular columns
        else:
            prices = prices_data
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Calculate risk metrics
    risk_metrics = data_loader.calculate_risk_metrics(returns, risk_free_rate=risk_free_rate)
    
    # Create output directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    risk_metrics.to_csv(os.path.join(output_dir, "risk_metrics.csv"))
    logger.info(f"Risk metrics saved to {os.path.join(output_dir, 'risk_metrics.csv')}")
    
    # Initialize optimizer
    optimizer = MarkowitzOptimizer(returns_data=returns, prices_data=prices)
    
    # Optimize for maximum Sharpe ratio
    weights_sharpe = optimizer.optimize_max_sharpe(risk_free_rate=risk_free_rate)
    
    # Get performance metrics for max Sharpe portfolio
    expected_return, expected_volatility, sharpe_ratio = optimizer.ef.portfolio_performance(risk_free_rate=risk_free_rate)
    
    logger.info("\nMaximum Sharpe Ratio Portfolio:")
    for ticker, weight in weights_sharpe.items():
        logger.info(f"{ticker}: {weight:.4f}")
    
    logger.info(f"Expected Return: {expected_return:.4f}")
    logger.info(f"Expected Volatility: {expected_volatility:.4f}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Optimize for minimum volatility
    weights_min_vol = optimizer.optimize_min_volatility()
    
    # Get performance metrics for min volatility portfolio
    expected_return_min_vol, expected_volatility_min_vol, sharpe_ratio_min_vol = optimizer.ef.portfolio_performance(risk_free_rate=risk_free_rate)
    
    logger.info("\nMinimum Volatility Portfolio:")
    for ticker, weight in weights_min_vol.items():
        logger.info(f"{ticker}: {weight:.4f}")
    
    logger.info(f"Expected Return: {expected_return_min_vol:.4f}")
    logger.info(f"Expected Volatility: {expected_volatility_min_vol:.4f}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio_min_vol:.4f}")
    
    # Save weights to CSV
    pd.DataFrame.from_dict(weights_sharpe, orient='index', columns=['Weight']).to_csv(
        os.path.join(output_dir, "max_sharpe_weights.csv")
    )
    pd.DataFrame.from_dict(weights_min_vol, orient='index', columns=['Weight']).to_csv(
        os.path.join(output_dir, "min_vol_weights.csv")
    )
    
    # Get discrete allocation
    latest_prices = prices.iloc[-1]
    portfolio_value = 10000
    
    # Discrete allocation for max Sharpe portfolio
    allocation_sharpe, leftover_sharpe = optimizer.get_discrete_allocation(
        weights_sharpe, total_portfolio_value=portfolio_value
    )
    
    # Discrete allocation for min volatility portfolio
    allocation_min_vol, leftover_min_vol = optimizer.get_discrete_allocation(
        weights_min_vol, total_portfolio_value=portfolio_value
    )
    
    logger.info(f"\nDiscrete Allocation (Max Sharpe) for ${portfolio_value} portfolio:")
    for ticker, shares in allocation_sharpe.items():
        logger.info(f"{ticker}: {shares} shares (${shares * latest_prices[ticker]:.2f})")
    logger.info(f"Remaining cash: ${leftover_sharpe:.2f}")
    
    logger.info(f"\nDiscrete Allocation (Min Volatility) for ${portfolio_value} portfolio:")
    for ticker, shares in allocation_min_vol.items():
        logger.info(f"{ticker}: {shares} shares (${shares * latest_prices[ticker]:.2f})")
    logger.info(f"Remaining cash: ${leftover_min_vol:.2f}")
    
    if plot:
        # Plot efficient frontier
        fig = optimizer.plot_efficient_frontier(risk_free_rate=risk_free_rate)
        fig.savefig(os.path.join(output_dir, "efficient_frontier.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Efficient frontier plot saved to {os.path.join(output_dir, 'efficient_frontier.png')}")
        
        # Plot correlation matrix
        fig = data_loader.plot_correlation_matrix(returns)
        fig.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Correlation matrix plot saved to {os.path.join(output_dir, 'correlation_matrix.png')}")
        
        # Plot prices
        fig = data_loader.plot_prices(prices)
        fig.savefig(os.path.join(output_dir, "prices.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Prices plot saved to {os.path.join(output_dir, 'prices.png')}")
        
        # Plot returns
        fig = data_loader.plot_returns(returns)
        fig.savefig(os.path.join(output_dir, "returns.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Returns plot saved to {os.path.join(output_dir, 'returns.png')}")
        
        # Plot rolling metrics
        fig = data_loader.plot_rolling_metrics(returns, risk_free_rate=risk_free_rate)
        fig.savefig(os.path.join(output_dir, "rolling_metrics.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Rolling metrics plot saved to {os.path.join(output_dir, 'rolling_metrics.png')}")
    
    # Return results
    results = {
        'max_sharpe': {
            'weights': weights_sharpe,
            'expected_return': expected_return,
            'expected_volatility': expected_volatility,
            'sharpe_ratio': sharpe_ratio,
            'allocation': allocation_sharpe,
            'leftover': leftover_sharpe
        },
        'min_volatility': {
            'weights': weights_min_vol,
            'expected_return': expected_return_min_vol,
            'expected_volatility': expected_volatility_min_vol,
            'sharpe_ratio': sharpe_ratio_min_vol,
            'allocation': allocation_min_vol,
            'leftover': leftover_min_vol
        },
        'risk_metrics': risk_metrics
    }
    
    return results

def run_monte_carlo_simulation(tickers, start_date, end_date=None, initial_investment=10000,
                              num_simulations=1000, num_periods=252, risk_free_rate=0.02,
                              output_dir="results", plot=True):
    """
    Run Monte Carlo simulation for portfolio performance.
    """
    logger.info(f"Running Monte Carlo simulation for {tickers}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load data
    prices_data = data_loader.load_yahoo_finance(tickers, start_date, end_date)
    
    # Check if 'Adj Close' is available, otherwise use 'Close'
    if 'Adj Close' in prices_data.columns:
        prices = prices_data['Adj Close']
    else:
        # If we have a MultiIndex with 'Close' at level 1
        if isinstance(prices_data.columns, pd.MultiIndex):
            prices = prices_data['Close']
        # If we just have regular columns
        else:
            prices = prices_data
    
    # Calculate returns
    returns = data_loader.calculate_returns(prices)
    
    # Run Markowitz optimization to get optimal weights
    optimizer = MarkowitzOptimizer(returns_data=returns, prices_data=prices)
    weights = optimizer.optimize_max_sharpe(risk_free_rate=risk_free_rate)
    
    # Initialize Monte Carlo simulator
    simulator = MonteCarloSimulator(
        returns_data=returns,
        weights=weights,
        initial_investment=initial_investment
    )
    
    # Run simulation
    simulation_results = simulator.run_simulation(
        num_simulations=num_simulations,
        num_periods=num_periods
    )
    
    # Calculate statistics
    stats = simulator.calculate_statistics()
    
    logger.info("\nMonte Carlo Simulation Results:")
    logger.info(f"Mean Final Portfolio Value: ${stats['mean_final_value']:.2f}")
    logger.info(f"Median Final Portfolio Value: ${stats['median_final_value']:.2f}")
    logger.info(f"Min Final Portfolio Value: ${stats['min_final_value']:.2f}")
    logger.info(f"Max Final Portfolio Value: ${stats['max_final_value']:.2f}")
    logger.info(f"5th Percentile: ${stats['percentile_5']:.2f}")
    logger.info(f"95th Percentile: ${stats['percentile_95']:.2f}")
    logger.info(f"Probability of Profit: {stats['prob_profit']:.2%}")
    
    # Save statistics to CSV
    pd.DataFrame([stats]).to_csv(os.path.join(output_dir, "monte_carlo_stats.csv"), index=False)
    
    if plot:
        # Plot simulation results
        fig = simulator.plot_simulation_results()
        fig.savefig(os.path.join(output_dir, "monte_carlo_simulation.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Monte Carlo simulation plot saved to {os.path.join(output_dir, 'monte_carlo_simulation.png')}")
        
        # Plot histogram of final values
        fig = simulator.plot_final_value_histogram()
        fig.savefig(os.path.join(output_dir, "monte_carlo_histogram.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Monte Carlo histogram plot saved to {os.path.join(output_dir, 'monte_carlo_histogram.png')}")
    
    # Return results
    results = {
        'weights': weights,
        'statistics': stats,
        'simulation_data': simulation_results
    }
    
    return results

def run_factor_model(target_asset, factor_tickers, start_date, end_date=None, 
                    model_type="random_forest", test_size=0.2, lookback_periods=5,
                    output_dir="results", plot=True):
    """
    Run factor model for risk assessment.
    """
    logger.info(f"Running factor model for {target_asset} using factors {factor_tickers}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load target asset data
    target_data = data_loader.load_yahoo_finance([target_asset], start_date, end_date)
    
    # Check if 'Adj Close' is available, otherwise use 'Close'
    if 'Adj Close' in target_data.columns:
        target_prices = target_data['Adj Close']
    else:
        # If we have a MultiIndex with 'Close' at level 1
        if isinstance(target_data.columns, pd.MultiIndex):
            target_prices = target_data['Close']
        # If we just have regular columns
        else:
            target_prices = target_data
            
    target_returns = data_loader.calculate_returns(target_prices)
    
    # Load factor data
    factor_data = data_loader.load_yahoo_finance(factor_tickers, start_date, end_date)
    
    # Check if 'Adj Close' is available, otherwise use 'Close'
    if 'Adj Close' in factor_data.columns:
        factor_prices = factor_data['Adj Close']
    else:
        # If we have a MultiIndex with 'Close' at level 1
        if isinstance(factor_data.columns, pd.MultiIndex):
            factor_prices = factor_data['Close']
        # If we just have regular columns
        else:
            factor_prices = factor_data
            
    factor_returns = data_loader.calculate_returns(factor_prices)
    
    # Initialize factor model
    factor_model = FactorModel(
        target_returns=target_returns,
        factor_returns=factor_returns,
        lookback_periods=lookback_periods
    )
    
    # Train model
    factor_model.train_model(model_type=model_type, test_size=test_size)
    
    # Evaluate model
    evaluation = factor_model.evaluate_model()
    
    logger.info("\nFactor Model Evaluation:")
    logger.info(f"R-squared: {evaluation['r2']:.4f}")
    logger.info(f"Mean Absolute Error: {evaluation['mae']:.4f}")
    logger.info(f"Root Mean Squared Error: {evaluation['rmse']:.4f}")
    
    # Save feature importances
    if model_type in ["random_forest", "xgboost"]:
        feature_importances = factor_model.get_feature_importances()
        pd.DataFrame(feature_importances, columns=['Importance']).to_csv(
            os.path.join(output_dir, "feature_importances.csv")
        )
        logger.info(f"Feature importances saved to {os.path.join(output_dir, 'feature_importances.csv')}")
    
    # Make predictions
    predictions = factor_model.predict()
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Actual': factor_model.y_test.values.flatten(),
        'Predicted': predictions.flatten()
    })
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    logger.info(f"Predictions saved to {os.path.join(output_dir, 'predictions.csv')}")
    
    if plot:
        # Plot actual vs predicted
        fig = factor_model.plot_actual_vs_predicted()
        fig.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Actual vs predicted plot saved to {os.path.join(output_dir, 'actual_vs_predicted.png')}")
        
        # Plot feature importances
        if model_type in ["random_forest", "xgboost"]:
            fig = factor_model.plot_feature_importances()
            fig.savefig(os.path.join(output_dir, "feature_importances.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Feature importances plot saved to {os.path.join(output_dir, 'feature_importances.png')}")
    
    # Return results
    results = {
        'evaluation': evaluation,
        'predictions': predictions_df,
        'model': factor_model.model
    }
    
    return results

def run_reinforcement_learning(tickers, start_date, end_date=None, risk_free_rate=0.02,
                              transaction_cost=0.001, window_size=30, max_steps=252,
                              num_iterations=100, output_dir="results", plot=True):
    """
    Train a reinforcement learning agent for dynamic portfolio allocation.
    """
    logger.info(f"Training reinforcement learning agent for {tickers}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load data
    prices_data = data_loader.load_yahoo_finance(tickers, start_date, end_date)
    
    # Check if 'Adj Close' is available, otherwise use 'Close'
    if 'Adj Close' in prices_data.columns:
        prices = prices_data['Adj Close']
    else:
        # If we have a MultiIndex with 'Close' at level 1
        if isinstance(prices_data.columns, pd.MultiIndex):
            prices = prices_data['Close']
        # If we just have regular columns
        else:
            prices = prices_data
    
    # Calculate returns
    returns = data_loader.calculate_returns(prices)
    
    # Split data into train and test sets
    train_size = int(len(returns) * 0.8)
    train_returns = returns.iloc[:train_size]
    test_returns = returns.iloc[train_size:]
    
    # Initialize RL optimizer
    rl_optimizer = RLPortfolioOptimizer(
        returns_data=train_returns,
        risk_free_rate=risk_free_rate,
        transaction_cost=transaction_cost,
        window_size=window_size,
        max_steps=max_steps
    )
    
    # Train the agent
    logger.info("Training RL agent (this may take a while)...")
    training_results = rl_optimizer.train(num_iterations=num_iterations)
    
    # Save training results
    pd.DataFrame(training_results).to_csv(os.path.join(output_dir, "rl_training_results.csv"), index=False)
    logger.info(f"Training results saved to {os.path.join(output_dir, 'rl_training_results.csv')}")
    
    # Backtest the trained agent
    logger.info("Backtesting RL agent...")
    backtest_results, weights_history = rl_optimizer.backtest(test_returns)
    
    # Save backtest results
    backtest_results.to_csv(os.path.join(output_dir, "rl_backtest_results.csv"))
    logger.info(f"Backtest results saved to {os.path.join(output_dir, 'rl_backtest_results.csv')}")
    
    # Save weights history
    weights_df = pd.DataFrame(weights_history, index=backtest_results.index)
    weights_df.to_csv(os.path.join(output_dir, "rl_weights_history.csv"))
    logger.info(f"Weights history saved to {os.path.join(output_dir, 'rl_weights_history.csv')}")
    
    if plot:
        # Plot backtest results
        fig = rl_optimizer.plot_backtest_results(backtest_results)
        fig.savefig(os.path.join(output_dir, "rl_backtest.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RL backtest plot saved to {os.path.join(output_dir, 'rl_backtest.png')}")
        
        # Plot weights history
        fig = rl_optimizer.plot_weights_history(weights_history, backtest_results.index)
        fig.savefig(os.path.join(output_dir, "rl_weights_history.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RL weights history plot saved to {os.path.join(output_dir, 'rl_weights_history.png')}")
        
        # Plot training progress
        fig = rl_optimizer.plot_training_progress(training_results)
        fig.savefig(os.path.join(output_dir, "rl_training_progress.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"RL training progress plot saved to {os.path.join(output_dir, 'rl_training_progress.png')}")
    
    # Return results
    results = {
        'training_results': training_results,
        'backtest_results': backtest_results,
        'weights_history': weights_history,
        'agent': rl_optimizer.agent
    }
    
    return results

def main():
    """
    Main function to run the portfolio optimization project.
    """
    parser = argparse.ArgumentParser(description="Portfolio Optimization")
    parser.add_argument("--mode", type=str, default="markowitz", 
                        choices=["markowitz", "monte_carlo", "factor", "reinforcement", "api"],
                        help="Mode to run")
    parser.add_argument("--tickers", type=str, nargs="+", default=["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
                        help="List of ticker symbols")
    parser.add_argument("--start_date", type=str, default="2018-01-01",
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, default=None,
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--risk_free_rate", type=float, default=0.02,
                        help="Risk-free rate (annualized)")
    parser.add_argument("--initial_investment", type=float, default=10000,
                        help="Initial investment amount")
    parser.add_argument("--num_simulations", type=int, default=1000,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--target_asset", type=str, default="SPY",
                        help="Target asset for factor model")
    parser.add_argument("--factor_tickers", type=str, nargs="+", 
                        default=["VTI", "BND", "GLD", "VNQ", "VWO"],
                        help="List of factor tickers")
    parser.add_argument("--model_type", type=str, default="random_forest",
                        choices=["random_forest", "linear", "xgboost"],
                        help="Model type for factor model")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--api_port", type=int, default=8000,
                        help="Port for FastAPI server")
    parser.add_argument("--no_plot", action="store_true",
                        help="Disable plotting")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected mode
    if args.mode == "markowitz":
        run_markowitz_optimization(
            args.tickers, 
            args.start_date, 
            args.end_date, 
            args.risk_free_rate,
            args.output_dir,
            not args.no_plot
        )
    elif args.mode == "monte_carlo":
        run_monte_carlo_simulation(
            args.tickers, 
            args.start_date, 
            args.end_date, 
            args.initial_investment, 
            args.num_simulations,
            252,  # One year of trading days
            args.risk_free_rate,
            args.output_dir,
            not args.no_plot
        )
    elif args.mode == "factor":
        run_factor_model(
            args.target_asset, 
            args.factor_tickers, 
            args.start_date, 
            args.end_date, 
            args.model_type,
            0.2,  # test_size
            5,    # lookback_periods
            args.output_dir,
            not args.no_plot
        )
    elif args.mode == "reinforcement":
        run_reinforcement_learning(
            args.tickers, 
            args.start_date, 
            args.end_date, 
            args.risk_free_rate,
            0.001,  # transaction_cost
            30,     # window_size
            252,    # max_steps
            100,    # num_iterations
            args.output_dir,
            not args.no_plot
        )
    elif args.mode == "api":
        try:
            import uvicorn
            from api.main import app
            logger.info(f"Starting FastAPI server on port {args.api_port}...")
            uvicorn.run("api.main:app", host="0.0.0.0", port=args.api_port, reload=True)
        except ImportError:
            logger.error("Failed to import FastAPI. Make sure it's installed.")
            sys.exit(1)

if __name__ == "__main__":
    main()

