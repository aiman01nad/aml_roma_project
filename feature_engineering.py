import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

flu_cleaned = pd.read_csv("dataset/flu.csv")

# Add temporal features
flu_cleaned['date'] = pd.to_datetime(flu_cleaned['date'])

# Extract temporal features
flu_cleaned['month'] = flu_cleaned['date'].dt.month
flu_cleaned['week'] = flu_cleaned['date'].dt.isocalendar().week
flu_cleaned['year'] = flu_cleaned['date'].dt.year
flu_cleaned['quarter'] = flu_cleaned['date'].dt.quarter
flu_cleaned['season'] = flu_cleaned['date'].dt.month % 12 // 3 + 1  # 1: Winter, 2: Spring, etc.

# Lag features for target and key predictors
lags = [1, 2, 4, 8]  # Lag intervals in weeks
for lag in lags:
    flu_cleaned[f'OT_lag{lag}'] = flu_cleaned['OT'].shift(lag)
    flu_cleaned[f'PERCENT_POSITIVE_lag{lag}'] = flu_cleaned['PERCENT POSITIVE'].shift(lag)
    flu_cleaned[f'TOTAL_A_lag{lag}'] = flu_cleaned['TOTAL A'].shift(lag)
    flu_cleaned[f'TOTAL_B_lag{lag}'] = flu_cleaned['TOTAL B'].shift(lag)

# Rolling averages
rolling_windows = [4, 8]  # 4-week and 8-week rolling windows
for window in rolling_windows:
    flu_cleaned[f'OT_roll_mean_{window}'] = flu_cleaned['OT'].rolling(window).mean()
    flu_cleaned[f'OT_roll_std_{window}'] = flu_cleaned['OT'].rolling(window).std()



# Final clean-up and filling missing values
flu_final = flu_cleaned.fillna(0)  # Or use another imputation method as earlier
flu_final.to_csv('dataset/flu_enhanced.csv', index=True)
