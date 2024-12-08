import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

flu_cleaned = pd.read_csv("dataset/flu.csv")

# Add temporal features
flu_cleaned['date'] = pd.to_datetime(flu_cleaned['date'])

flu_cleaned['month'] = flu_cleaned['date'].dt.month
flu_cleaned['week'] = flu_cleaned['date'].dt.isocalendar().week
flu_cleaned['year'] = flu_cleaned['date'].dt.year
flu_cleaned['quarter'] = flu_cleaned['date'].dt.quarter
flu_cleaned['season'] = flu_cleaned['date'].dt.month % 12 // 3 + 1  # 1: Winter, 2: Spring, etc.

""" # Lag features for target and key predictors
lags = [1, 2, 4, 8]  # Lag intervals in weeks
for lag in lags:
    flu_cleaned[f'OT_lag{lag}'] = flu_cleaned['OT'].shift(lag)
    flu_cleaned[f'PERCENT_POSITIVE_lag{lag}'] = flu_cleaned['PERCENT POSITIVE'].shift(lag)
    flu_cleaned[f'TOTAL_A_lag{lag}'] = flu_cleaned['TOTAL A'].shift(lag)
    flu_cleaned[f'TOTAL_B_lag{lag}'] = flu_cleaned['TOTAL B'].shift(lag) """

# Rolling averages
rolling_windows = [4, 8]  # 4-week and 8-week rolling windows
for window in rolling_windows:
    flu_cleaned[f'OT_roll_mean_{window}'] = flu_cleaned['OT'].rolling(window).mean()
    flu_cleaned[f'OT_roll_std_{window}'] = flu_cleaned['OT'].rolling(window).std()

# Data normalization
train_end = 849  # 1997-2013
val_end = 1006  # 2014-2016
test_end = 1162  # 2017-2019
seq_len = 156

# Split the data
train_df = flu_cleaned[:train_end]
val_df = flu_cleaned[train_end - seq_len:val_end]
test_df = flu_cleaned[val_end - seq_len:test_end]

target_column = 'OT'
X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]
X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

# Keep track of the temporal features
temporal_features = ['date', 'month', 'week', 'year', 'quarter', 'season']
temporal_train = train_df[temporal_features]
temporal_val = val_df[temporal_features]
temporal_test = test_df[temporal_features]

# Drop the temporal features from the feature set for scaling
X_train = X_train.drop(columns=temporal_features)
X_val = X_val.drop(columns=temporal_features)
X_test = X_test.drop(columns=temporal_features)

# Scale the feature columns
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Scale the target column
target_scaler = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

# Add overlapping sequences for validation and test
val_seq_scaled = X_train_scaled.iloc[-seq_len:]  # Last `seq_len` rows of training set
X_val_scaled = pd.concat([val_seq_scaled, X_val_scaled], axis=0).reset_index(drop=True)

test_seq_scaled = X_val_scaled.iloc[-seq_len:]  # Last `seq_len` rows of validation set
X_test_scaled = pd.concat([test_seq_scaled, X_test_scaled], axis=0).reset_index(drop=True)

# Combine scaled features and targets
train_scaled = pd.concat([X_train_scaled, pd.Series(y_train_scaled.flatten(), name=target_column)], axis=1)
val_scaled = pd.concat([X_val_scaled, pd.Series(y_val_scaled.flatten(), name=target_column)], axis=1)
test_scaled = pd.concat([X_test_scaled, pd.Series(y_test_scaled.flatten(), name=target_column)], axis=1)

# Add dates back to the scaled data
train_scaled_with_date = pd.concat([temporal_train.reset_index(drop=True), train_scaled.reset_index(drop=True)], axis=1)
val_scaled_with_date = pd.concat([temporal_val.reset_index(drop=True), val_scaled.reset_index(drop=True)], axis=1)
test_scaled_with_date = pd.concat([temporal_test.reset_index(drop=True), test_scaled.reset_index(drop=True)], axis=1)

# Combine all data
flu_cleaned = pd.concat([train_scaled_with_date, val_scaled_with_date, test_scaled_with_date], axis=0)

# Verify flu_final
print(flu_cleaned.head())
print(flu_cleaned.tail())

# Final clean-up and filling missing values
flu_final = flu_cleaned.fillna(0)
flu_final.to_csv('dataset/flu_enhanced.csv', index=False)
