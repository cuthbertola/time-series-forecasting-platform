"""Time series feature engineering."""
import numpy as np
import pandas as pd
from typing import Tuple, List


class TimeSeriesFeatureEngineer:
    """Create features for time series forecasting."""
    
    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        self.feature_names = []
    
    def create_features(
        self, 
        df: pd.DataFrame, 
        date_column: str = 'date', 
        target_column: str = 'sales'
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create features from time series data.
        
        Args:
            df: DataFrame with date and target columns
            date_column: Name of date column
            target_column: Name of target column
        
        Returns:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        
        features = []
        feature_names = []
        
        # Lag features
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            if lag < len(df):
                col_name = f'lag_{lag}'
                df[col_name] = df[target_column].shift(lag)
                features.append(col_name)
                feature_names.append(col_name)
        
        # Rolling statistics
        for window in [7, 14, 28]:
            if window < len(df):
                # Rolling mean
                col_name = f'rolling_mean_{window}'
                df[col_name] = df[target_column].shift(1).rolling(window=window).mean()
                features.append(col_name)
                feature_names.append(col_name)
                
                # Rolling std
                col_name = f'rolling_std_{window}'
                df[col_name] = df[target_column].shift(1).rolling(window=window).std()
                features.append(col_name)
                feature_names.append(col_name)
        
        # Date features
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['day_of_month'] = df[date_column].dt.day
        df['month'] = df[date_column].dt.month
        df['week_of_year'] = df[date_column].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df[date_column].dt.dayofweek >= 5).astype(int)
        
        date_features = ['day_of_week', 'day_of_month', 'month', 'week_of_year', 'is_weekend']
        features.extend(date_features)
        feature_names.extend(date_features)
        
        # Drop NaN rows (from lag/rolling features)
        df_clean = df.dropna(subset=features + [target_column])
        
        X = df_clean[features].values
        y = df_clean[target_column].values
        
        self.feature_names = feature_names
        self.lookback = len(df) - len(df_clean)
        
        return X, y, feature_names
    
    def create_future_features(
        self,
        last_values: np.ndarray,
        last_date: pd.Timestamp,
        horizon: int = 30
    ) -> Tuple[np.ndarray, List[pd.Timestamp]]:
        """
        Create features for future predictions.
        
        Args:
            last_values: Recent target values (at least 28 days)
            last_date: Last date in the training data
            horizon: Number of days to forecast
        
        Returns:
            X_future: Feature matrix for future dates
            future_dates: List of future dates
        """
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]
        future_features = []
        
        values = list(last_values)
        
        for i, date in enumerate(future_dates):
            features = []
            
            # Lag features (using values array which grows with predictions)
            for lag in [1, 2, 3, 7, 14, 21, 28]:
                if len(values) >= lag:
                    features.append(values[-lag])
                else:
                    features.append(values[0])
            
            # Rolling statistics
            for window in [7, 14, 28]:
                recent = values[-window:] if len(values) >= window else values
                features.append(np.mean(recent))  # rolling mean
                features.append(np.std(recent) if len(recent) > 1 else 0)  # rolling std
            
            # Date features
            features.append(date.dayofweek)  # day_of_week
            features.append(date.day)  # day_of_month
            features.append(date.month)  # month
            features.append(date.isocalendar()[1])  # week_of_year
            features.append(1 if date.dayofweek >= 5 else 0)  # is_weekend
            
            future_features.append(features)
            
            # Placeholder for the predicted value (will be updated during forecasting)
            values.append(values[-1])  # Use last value as placeholder
        
        return np.array(future_features), future_dates
