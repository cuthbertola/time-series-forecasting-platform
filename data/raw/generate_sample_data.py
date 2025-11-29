import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates for 2 years of daily data
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate realistic sales data with:
# - Trend: gradual increase over time
# - Yearly seasonality: higher in Q4 (holiday season)
# - Weekly seasonality: higher on weekends
# - Random noise

n_days = len(dates)

# Base sales
base_sales = 1000

# Trend component (gradual increase)
trend = np.linspace(0, 300, n_days)

# Yearly seasonality (higher in Q4)
day_of_year = np.array([d.timetuple().tm_yday for d in dates])
yearly_seasonality = 200 * np.sin(2 * np.pi * (day_of_year - 60) / 365)

# Weekly seasonality (higher on weekends)
day_of_week = np.array([d.weekday() for d in dates])
weekly_seasonality = np.where(day_of_week >= 5, 150, 0)  # Weekend boost

# Monthly pattern (end of month boost)
day_of_month = np.array([d.day for d in dates])
monthly_pattern = np.where(day_of_month > 25, 100, 0)

# Holiday effects (Christmas, Black Friday, etc.)
holiday_boost = np.zeros(n_days)
for i, date in enumerate(dates):
    # Christmas period (Dec 15-25)
    if date.month == 12 and 15 <= date.day <= 25:
        holiday_boost[i] = 500
    # Black Friday period (late November)
    elif date.month == 11 and 20 <= date.day <= 30:
        holiday_boost[i] = 400
    # New Year
    elif date.month == 1 and date.day <= 5:
        holiday_boost[i] = 200

# Random noise
noise = np.random.normal(0, 50, n_days)

# Combine all components
sales = base_sales + trend + yearly_seasonality + weekly_seasonality + monthly_pattern + holiday_boost + noise
sales = np.maximum(sales, 100)  # Ensure no negative sales

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'sales': np.round(sales, 2),
    'day_of_week': day_of_week,
    'month': [d.month for d in dates],
    'is_weekend': (day_of_week >= 5).astype(int),
    'temperature': np.random.normal(60, 15, n_days),  # Weather feature
    'promotion': np.random.binomial(1, 0.1, n_days)   # Random promotions
})

# Save to CSV
df.to_csv('sample_sales_data.csv', index=False)
print(f"Generated sample dataset with {len(df)} rows")
print(f"\nDataset preview:")
print(df.head(10))
print(f"\nDataset statistics:")
print(df.describe())
print(f"\nSaved to: data/raw/sample_sales_data.csv")
