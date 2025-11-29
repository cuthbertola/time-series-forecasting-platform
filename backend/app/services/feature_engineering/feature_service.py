import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime
import holidays


class TimeSeriesFeatureEngineer:
    """Automated feature engineering for time series data."""
    
    def __init__(self, df: pd.DataFrame, date_column: str, target_column: str):
        self.df = df.copy()
        self.date_column = date_column
        self.target_column = target_column
        self.created_features: List[str] = []
        
        # Ensure date column is datetime
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.df = self.df.sort_values(date_column).reset_index(drop=True)
    
    def create_all_features(
        self,
        lag_periods: List[int] = [1, 7, 14, 30],
        rolling_windows: List[int] = [7, 14, 30],
        include_calendar: bool = True,
        include_trend: bool = True,
        country_code: str = "US"
    ) -> pd.DataFrame:
        """Create all time series features."""
        
        self.create_lag_features(lag_periods)
        self.create_rolling_features(rolling_windows)
        
        if include_calendar:
            self.create_calendar_features(country_code)
        
        if include_trend:
            self.create_trend_features()
        
        return self.df
    
    def create_lag_features(self, periods: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """Create lag features for the target variable."""
        for period in periods:
            feature_name = f"lag_{period}"
            self.df[feature_name] = self.df[self.target_column].shift(period)
            self.created_features.append(feature_name)
        
        return self.df
    
    def create_rolling_features(self, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """Create rolling statistics features."""
        for window in windows:
            # Rolling mean
            feature_name = f"rolling_mean_{window}"
            self.df[feature_name] = self.df[self.target_column].rolling(window=window).mean()
            self.created_features.append(feature_name)
            
            # Rolling std
            feature_name = f"rolling_std_{window}"
            self.df[feature_name] = self.df[self.target_column].rolling(window=window).std()
            self.created_features.append(feature_name)
            
            # Rolling min
            feature_name = f"rolling_min_{window}"
            self.df[feature_name] = self.df[self.target_column].rolling(window=window).min()
            self.created_features.append(feature_name)
            
            # Rolling max
            feature_name = f"rolling_max_{window}"
            self.df[feature_name] = self.df[self.target_column].rolling(window=window).max()
            self.created_features.append(feature_name)
        
        return self.df
    
    def create_calendar_features(self, country_code: str = "US") -> pd.DataFrame:
        """Create calendar-based features."""
        dates = self.df[self.date_column]
        
        # Basic calendar features
        self.df["day_of_week"] = dates.dt.dayofweek
        self.df["day_of_month"] = dates.dt.day
        self.df["day_of_year"] = dates.dt.dayofyear
        self.df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        self.df["month"] = dates.dt.month
        self.df["quarter"] = dates.dt.quarter
        self.df["year"] = dates.dt.year
        
        # Weekend indicator
        self.df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
        
        # Month start/end indicators
        self.df["is_month_start"] = dates.dt.is_month_start.astype(int)
        self.df["is_month_end"] = dates.dt.is_month_end.astype(int)
        
        # Quarter start/end indicators
        self.df["is_quarter_start"] = dates.dt.is_quarter_start.astype(int)
        self.df["is_quarter_end"] = dates.dt.is_quarter_end.astype(int)
        
        # Holiday features
        try:
            country_holidays = holidays.country_holidays(country_code)
            self.df["is_holiday"] = dates.apply(lambda x: 1 if x in country_holidays else 0)
        except Exception:
            self.df["is_holiday"] = 0
        
        calendar_features = [
            "day_of_week", "day_of_month", "day_of_year", "week_of_year",
            "month", "quarter", "year", "is_weekend", "is_month_start",
            "is_month_end", "is_quarter_start", "is_quarter_end", "is_holiday"
        ]
        self.created_features.extend(calendar_features)
        
        return self.df
    
    def create_trend_features(self) -> pd.DataFrame:
        """Create trend-based features."""
        # Time index (days since start)
        min_date = self.df[self.date_column].min()
        self.df["days_since_start"] = (self.df[self.date_column] - min_date).dt.days
        
        # Cyclical encoding for periodic patterns
        self.df["day_sin"] = np.sin(2 * np.pi * self.df["day_of_year"] / 365)
        self.df["day_cos"] = np.cos(2 * np.pi * self.df["day_of_year"] / 365)
        self.df["week_sin"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["week_cos"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)
        
        trend_features = [
            "days_since_start", "day_sin", "day_cos",
            "week_sin", "week_cos", "month_sin", "month_cos"
        ]
        self.created_features.extend(trend_features)
        
        return self.df
    
    def get_feature_names(self) -> List[str]:
        """Return list of created feature names."""
        return self.created_features
    
    def detect_frequency(self) -> str:
        """Detect the frequency of the time series."""
        dates = self.df[self.date_column].sort_values()
        diffs = dates.diff().dropna()
        median_diff = diffs.median()
        
        if median_diff <= pd.Timedelta(hours=1):
            return "hourly"
        elif median_diff <= pd.Timedelta(days=1):
            return "daily"
        elif median_diff <= pd.Timedelta(days=7):
            return "weekly"
        elif median_diff <= pd.Timedelta(days=31):
            return "monthly"
        elif median_diff <= pd.Timedelta(days=92):
            return "quarterly"
        else:
            return "yearly"
    
    def detect_seasonality(self) -> dict:
        """Detect potential seasonality patterns."""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        try:
            # Set date as index
            ts = self.df.set_index(self.date_column)[self.target_column]
            
            # Determine period based on frequency
            freq = self.detect_frequency()
            period_map = {
                "daily": 7,
                "weekly": 52,
                "monthly": 12,
                "quarterly": 4,
                "yearly": 1
            }
            period = period_map.get(freq, 7)
            
            if len(ts) < 2 * period:
                return {"has_seasonality": False, "period": None}
            
            decomposition = seasonal_decompose(ts, period=period, extrapolate_trend="freq")
            seasonal_strength = decomposition.seasonal.var() / (decomposition.seasonal.var() + decomposition.resid.var())
            
            return {
                "has_seasonality": seasonal_strength > 0.1,
                "seasonal_strength": float(seasonal_strength),
                "period": period,
                "frequency": freq
            }
        except Exception as e:
            return {"has_seasonality": False, "error": str(e)}
