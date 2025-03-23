"""
Configuration settings for the portfolio optimization project.
"""

# Default tickers for portfolio optimization
DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK-B", "JPM", "JNJ"]

# Default date range
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = None  # Use current date

# Risk parameters
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annualized
DEFAULT_TRANSACTION_COST = 0.001  # 0.1% per trade

# Monte Carlo simulation parameters
DEFAULT_INITIAL_INVESTMENT = 10000
DEFAULT_NUM_SIMULATIONS = 1000
DEFAULT_SIMULATION_PERIODS = 252  # One trading year

# Factor model parameters
DEFAULT_TARGET_ASSET = "SPY"
DEFAULT_FACTOR_TICKERS = ["VTI", "BND", "GLD", "VNQ", "VWO"]
DEFAULT_LOOKBACK_PERIODS = 5
DEFAULT_TEST_SIZE = 0.2

# Reinforcement learning parameters
DEFAULT_WINDOW_SIZE = 30
DEFAULT_MAX_STEPS = 252
DEFAULT_NUM_ITERATIONS = 100

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# Data directory
DATA_DIR = "data"

# Visualization settings
PLOT_DPI = 300
PLOT_FIGSIZE = (12, 8)
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# Colors for plotting
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "tertiary": "#2ca02c",
    "quaternary": "#d62728",
    "quinary": "#9467bd",
    "senary": "#8c564b",
    "septenary": "#e377c2",
    "octonary": "#7f7f7f",
    "nonary": "#bcbd22",
    "denary": "#17becf"
}

# Benchmark tickers
BENCHMARK_TICKERS = {
    "US_MARKET": "SPY",
    "GLOBAL_MARKET": "ACWI",
    "BONDS": "AGG",
    "GOLD": "GLD",
    "REAL_ESTATE": "VNQ"
}

# Factor model features
FACTOR_FEATURES = {
    "MARKET": "SPY",  # Market factor
    "SIZE": "IWM",    # Small-cap (size factor)
    "VALUE": "IWD",   # Value factor
    "MOMENTUM": "MTUM",  # Momentum factor
    "QUALITY": "QUAL",   # Quality factor
    "VOLATILITY": "USMV"  # Low volatility factor
}

# Optimization constraints
OPTIMIZATION_CONSTRAINTS = {
    "MAX_WEIGHT_PER_ASSET": 0.3,  # Maximum 30% in any single asset
    "MIN_WEIGHT_PER_ASSET": 0.05,  # Minimum 5% in any included asset
    "MAX_SECTOR_EXPOSURE": 0.4     # Maximum 40% in any single sector
}