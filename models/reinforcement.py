import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
import gymnasium as gym
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class PortfolioEnv(gym.Env):
    """
    Portfolio optimization environment for reinforcement learning.
    """
    
    def __init__(self, config=None):
        """
        Initialize the environment.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with the following keys:
            - returns_data: pandas.DataFrame with asset returns
            - risk_free_rate: float, risk-free rate
            - transaction_cost: float, transaction cost as a fraction of trade value
            - window_size: int, number of days to include in state
            - max_steps: int, maximum number of steps per episode
        """
        super(PortfolioEnv, self).__init__()
        
        # Set default config if not provided
        if config is None:
            config = {}
            
        # Get config parameters
        self.returns_data = config.get('returns_data', None)
        self.risk_free_rate = config.get('risk_free_rate', 0.0)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.window_size = config.get('window_size', 30)
        self.max_steps = config.get('max_steps', 252)  # One trading year
        
        # Check if returns data is provided
        if self.returns_data is None:
            raise ValueError("Returns data must be provided in config")
            
        # Set number of assets
        self.num_assets = self.returns_data.shape[1]
        
        # Define action space (portfolio weights)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_assets,), dtype=np.float32
        )
        
        # Define observation space (returns history + current portfolio weights)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size * self.num_assets + self.num_assets,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Returns:
        --------
        numpy.ndarray
            Initial observation
        dict
            Info dictionary
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Initialize portfolio (equal weights)
        self.portfolio_weights = np.ones(self.num_assets) / self.num_assets
        
        # Initialize portfolio value
        self.portfolio_value = 1.0
        
        # Initialize returns history
        self.returns_history = []
        
        # Choose random starting point
        self.start_idx = np.random.randint(
            self.window_size, len(self.returns_data) - self.max_steps
        )
        
        # Get initial observation
        observation = self._get_observation()
        
        # Return observation and info dict
        return observation, {}
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        action : numpy.ndarray
            Portfolio weights
            
        Returns:
        --------
        numpy.ndarray
            Next observation
        float
            Reward
        bool
            Whether the episode is done
        bool
            Whether the episode is truncated
        dict
            Info dictionary
        """
        # Normalize action to ensure weights sum to 1
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            # If all actions are zero or negative, use equal weights
            action = np.ones_like(action) / len(action)
        
        # Calculate transaction costs
        transaction_cost = np.sum(np.abs(action - self.portfolio_weights)) * self.transaction_cost
        
        # Update portfolio weights
        self.portfolio_weights = action
        
        # Get returns for current step
        current_returns = self.returns_data.iloc[self.start_idx + self.current_step].values
        
        # Calculate portfolio return
        portfolio_return = np.sum(current_returns * self.portfolio_weights) - transaction_cost
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        
        # Store returns for observation
        self.returns_history.append(current_returns)
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)
            
        # Increment step counter
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get next observation
        observation = self._get_observation()
        
        # Calculate reward (Sharpe ratio for this step)
        reward = self._calculate_reward(portfolio_return)
        
        return observation, reward, done, False, {'portfolio_value': self.portfolio_value}
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
        --------
        numpy.ndarray
            Current observation
        """
        # If returns history is not full yet, pad with zeros
        if len(self.returns_history) < self.window_size:
            padded_history = [np.zeros(self.num_assets)] * (self.window_size - len(self.returns_history))
            padded_history.extend(self.returns_history)
        else:
            padded_history = self.returns_history
            
        # Flatten returns history
        flattened_history = np.array(padded_history).flatten()
        
        # Concatenate with current portfolio weights
        observation = np.concatenate([flattened_history, self.portfolio_weights])
        
        return observation
    
    def _calculate_reward(self, portfolio_return):
        """
        Calculate the reward for the current step.
        
        Parameters:
        -----------
        portfolio_return : float
            Portfolio return for the current step
            
        Returns:
        --------
        float
            Reward
        """
        # Use Sharpe ratio as reward
        excess_return = portfolio_return - self.risk_free_rate / 252  # Daily risk-free rate
        
        # If we have enough history, calculate rolling volatility
        if len(self.returns_history) >= 20:  # At least 20 days for meaningful volatility
            portfolio_returns = np.array([np.sum(r * self.portfolio_weights) for r in self.returns_history[-20:]])
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            
            # Avoid division by zero
            if volatility > 0:
                sharpe = excess_return / volatility
            else:
                sharpe = excess_return  # If volatility is zero, just use excess return
        else:
            # Not enough history, use excess return as reward
            sharpe = excess_return
            
        return sharpe


class RLPortfolioOptimizer:
    def __init__(self, returns_data, risk_free_rate=0.02, transaction_cost=0.001,
                window_size=30, max_steps=252):
        """
        Initialize the RL portfolio optimizer.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame
            DataFrame with asset returns
        risk_free_rate : float, default=0.02
            Risk-free rate (annualized)
        transaction_cost : float, default=0.001
            Transaction cost as a fraction of trade value
        window_size : int, default=30
            Number of days to include in state
        max_steps : int, default=252
            Maximum number of steps per episode
        """
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.max_steps = max_steps
        self.num_assets = returns_data.shape[1]
        self.model = None
        
        # Environment configuration
        self.env_config = {
            'returns_data': returns_data,
            'risk_free_rate': risk_free_rate,
            'transaction_cost': transaction_cost,
            'window_size': window_size,
            'max_steps': max_steps
        }
        
    def _make_env(self):
        """
        Create environment for stable-baselines3.
        
        Returns:
        --------
        gym.Env
            Portfolio environment
        """
        return PortfolioEnv(self.env_config)
    
    def train(self, num_iterations=100, checkpoint_freq=10, checkpoint_dir='checkpoints'):
        """
        Train the RL agent.
        
        Parameters:
        -----------
        num_iterations : int, default=100
            Number of training iterations
        checkpoint_freq : int, default=10
            Frequency of checkpoints
        checkpoint_dir : str, default='checkpoints'
            Directory to save checkpoints
            
        Returns:
        --------
        list
            Training results
        """
        # Create vectorized environment
        env = DummyVecEnv([self._make_env])
        
        # Create PPO agent
        self.model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=5e-5,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0
        )
        
        # Train the agent
        results = []
        total_timesteps = num_iterations * self.max_steps
        
        # Use tqdm for progress tracking
        with tqdm(total=total_timesteps, desc="Training RL agent") as pbar:
            for i in range(num_iterations):
                # Train for one iteration
                self.model.learn(total_timesteps=self.max_steps, reset_num_timesteps=False, progress_bar=False)
                
                # Update progress bar
                pbar.update(self.max_steps)
                
                # Save checkpoint
                if (i + 1) % checkpoint_freq == 0:
                    self.model.save(f"{checkpoint_dir}/ppo_portfolio_{i+1}")
                    print(f"Checkpoint saved at iteration {i+1}")
                
                # Evaluate model
                eval_env = self._make_env()
                obs, _ = eval_env.reset()
                done = False
                rewards = []
                portfolio_value = 1.0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = eval_env.step(action)
                    rewards.append(reward)
                    portfolio_value = info['portfolio_value']
                
                # Store results
                results.append({
                    'iteration': i + 1,
                    'mean_reward': np.mean(rewards),
                    'final_portfolio_value': portfolio_value
                })
                
                print(f"Iteration {i+1}/{num_iterations} - Mean Reward: {np.mean(rewards):.4f} - Final Value: {portfolio_value:.4f}")
        
        return results
    
    def backtest(self, test_returns, initial_value=1.0):
        """
        Backtest the trained RL agent.
        
        Parameters:
        -----------
        test_returns : pandas.DataFrame
            DataFrame with test returns
        initial_value : float, default=1.0
            Initial portfolio value
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with backtest results
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        # Create environment for testing
        env_config = self.env_config.copy()
        env_config['returns_data'] = test_returns
        env_config['max_steps'] = len(test_returns)
        
        env = PortfolioEnv(env_config)
        
        # Initialize backtest
        obs, _ = env.reset()
        done = False
        portfolio_values = [initial_value]
        weights_history = []
        
        # Run backtest
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            portfolio_values.append(info['portfolio_value'] * initial_value)
            weights_history.append(env.portfolio_weights)
            
        # Create results DataFrame
        results = pd.DataFrame({
            'portfolio_value': portfolio_values
        }, index=test_returns.index[:len(portfolio_values)])
        
        # Add weights history
        weights_df = pd.DataFrame(
            weights_history, 
            columns=test_returns.columns,
            index=test_returns.index[:len(weights_history)]
        )
        
        return results, weights_df
    
    def plot_backtest_results(self, results, benchmark=None, figsize=(12, 8)):
        """
        Plot backtest results.
        
        Parameters:
        -----------
        results : pandas.DataFrame
            DataFrame with backtest results
        benchmark : pandas.DataFrame, optional
            DataFrame with benchmark results
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot portfolio value
        ax.plot(results.index, results['portfolio_value'], label='RL Portfolio')
        
        # Plot benchmark if provided
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark, label='Benchmark')
            
        # Add labels and title
        ax.set_title('RL Portfolio Backtest Results', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig
        
    def plot_weights_history(self, weights_df, figsize=(15, 8)):
        """
        Plot weights history.
        
        Parameters:
        -----------
        weights_df : pandas.DataFrame
            DataFrame with weights history
        figsize : tuple, default=(15, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot weights as area chart
        weights_df.plot(kind='area', stacked=True, ax=ax)
        
        # Add labels and title
        ax.set_title('Portfolio Weights Over Time', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig