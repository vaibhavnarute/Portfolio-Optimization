import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Try to import XGBoost, but make it optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. XGBoost models will not be available.")

class FactorModel:
    """
    Factor model for risk assessment.
    """
    
    def __init__(self, target_returns, factor_returns, lookback_periods=5):
        """
        Initialize the factor model.
        
        Parameters:
        -----------
        target_returns : pandas.DataFrame
            DataFrame with target asset returns
        factor_returns : pandas.DataFrame
            DataFrame with factor returns
        lookback_periods : int, default=5
            Number of periods to use for lookback features
        """
        self.target_returns = target_returns
        self.factor_returns = factor_returns
        self.lookback_periods = lookback_periods
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def _prepare_data(self):
        """
        Prepare data for model training.
        
        Returns:
        --------
        tuple
            X and y data
        """
        # Align data
        aligned_data = pd.concat([self.target_returns, self.factor_returns], axis=1).dropna()
        
        # Extract target and factors
        target = aligned_data.iloc[:, 0]
        factors = aligned_data.iloc[:, 1:]
        
        # Create lagged features
        X = pd.DataFrame()
        
        for lag in range(1, self.lookback_periods + 1):
            lagged_factors = factors.shift(lag)
            lagged_factors.columns = [f"{col}_lag{lag}" for col in factors.columns]
            X = pd.concat([X, lagged_factors], axis=1)
        
        # Add current factors
        X = pd.concat([X, factors], axis=1)
        
        # Drop rows with NaN values
        X = X.dropna()
        y = target.loc[X.index]
        
        # Store feature names
        self.feature_names = X.columns
        
        return X, y
    
    def train_model(self, model_type="random_forest", test_size=0.2, random_state=42):
        """
        Train the factor model.
        
        Parameters:
        -----------
        model_type : str, default="random_forest"
            Type of model to use (random_forest, linear, xgboost)
        test_size : float, default=0.2
            Proportion of data to use for testing
        random_state : int, default=42
            Random state for reproducibility
            
        Returns:
        --------
        object
            Trained model
        """
        # Prepare data
        X, y = self._prepare_data()
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Initialize model
        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not installed. Please install it with 'pip install xgboost'")
            self.model = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
        else:
            raise ValueError("Model type must be 'linear', 'random_forest', or 'xgboost'")
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate the model.
        
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }
    
    def predict(self, X=None):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : pandas.DataFrame, optional
            Data to predict on (if None, use test data)
            
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        if X is None:
            X = self.X_test
        
        return self.model.predict(X)
    
    def get_feature_importances(self):
        """
        Get feature importances.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature importances")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create DataFrame
        feature_importances = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importances = feature_importances.sort_values('Importance', ascending=False)
        
        return feature_importances
    
    def plot_actual_vs_predicted(self, figsize=(12, 8)):
        """
        Plot actual vs predicted values.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual vs predicted
        ax.scatter(self.y_test, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add labels and title
        ax.set_title('Actual vs Predicted Returns', fontsize=14)
        ax.set_xlabel('Actual Returns', fontsize=12)
        ax.set_ylabel('Predicted Returns', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig
    
    def plot_feature_importances(self, top_n=10, figsize=(12, 8)):
        """
        Plot feature importances.
        
        Parameters:
        -----------
        top_n : int, default=10
            Number of top features to plot
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature importances")
        
        # Get feature importances
        feature_importances = self.get_feature_importances()
        
        # Select top N features
        top_features = feature_importances.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot feature importances
        ax.barh(top_features['Feature'], top_features['Importance'])
        
        # Add labels and title
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig
    
    def plot_prediction_time_series(self, figsize=(15, 8)):
        """
        Plot time series of actual vs predicted values.
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Create DataFrame with actual and predicted values
        pred_df = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': y_pred
        }, index=self.y_test.index)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot time series
        pred_df.plot(ax=ax)
        
        # Add labels and title
        ax.set_title('Actual vs Predicted Returns Over Time', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Returns', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return fig
    
    def plot_residuals(self, figsize=(12, 8)):
        """
        Plot residuals.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.model is None:
            raise ValueError("Train model first")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate residuals
        residuals = self.y_test - y_pred
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot residuals vs predicted
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('Residuals vs Predicted', fontsize=12)
        ax1.set_xlabel('Predicted Returns', fontsize=10)
        ax1.set_ylabel('Residuals', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Plot residual histogram
        ax2.hist(residuals, bins=30, alpha=0.7)
        ax2.set_title('Residual Distribution', fontsize=12)
        ax2.set_xlabel('Residual', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Adjust layout
        fig.tight_layout()
        
        return fig