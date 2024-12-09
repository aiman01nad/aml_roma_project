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
train_end = 847  # 1997-2013
val_end = 1005  # 2014-2016
test_end = 1161  # 2017-2019

# Split the data
train_df = flu_cleaned.iloc[:train_end]
val_df = flu_cleaned.iloc[train_end+1:val_end]
test_df = flu_cleaned.iloc[val_end+1:test_end]

target = 'OT'
X_train, y_train = train_df.drop(columns=[target]), train_df[target]
X_val, y_val = val_df.drop(columns=[target]), val_df[target]
X_test, y_test = test_df.drop(columns=[target]), test_df[target]

# Don't scale the temporal features
temporal_features = ['date', 'month', 'week', 'year', 'quarter', 'season']
temporal_train = train_df[temporal_features]
temporal_val = val_df[temporal_features]
temporal_test = test_df[temporal_features]

X_train = X_train.drop(columns=temporal_features)
X_val = X_val.drop(columns=temporal_features)
X_test = X_test.drop(columns=temporal_features)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

target_scaler = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

# Add target and temporal features back to the scaled data
X_train_scaled[target] = y_train_scaled
X_val_scaled[target] = y_val_scaled
X_test_scaled[target] = y_test_scaled

X_train_scaled[temporal_features] = temporal_train[temporal_features].reset_index(drop=True)
X_val_scaled[temporal_features] = temporal_val[temporal_features].reset_index(drop=True)
X_test_scaled[temporal_features] = temporal_test[temporal_features].reset_index(drop=True)

# Combine all data
flu_cleaned = pd.concat([X_train_scaled, X_val_scaled, X_test_scaled], axis=0)
columns = ['date'] + [col for col in flu_cleaned.columns if col != 'date']
flu_cleaned = flu_cleaned[columns]

# Verify flu_final
#print(flu_cleaned.head())
#print(flu_cleaned.tail())

# Final clean-up and filling missing values
flu_final = flu_cleaned.fillna(0)
flu_final.to_csv('dataset/flu_enhanced.csv', index=False)

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(flu_final['date'], flu_final['OT'], label='Weighted ILI')
plt.plot(flu_final['date'], flu_final['OT_roll_mean_4'], label='4-week rolling mean')
plt.plot(flu_final['date'], flu_final['OT_roll_mean_8'], label='8-week rolling mean')
plt.title('Observed vs. Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Observed mean')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(flu_final['date'], flu_final['TOTAL SPECIMENS'], label='Total Specimens')
plt.plot(flu_final['date'], flu_final['TOTAL A'], label='Total A')
plt.plot(flu_final['date'], flu_final['TOTAL B'], label='Total B')
plt.plot(flu_final['date'], flu_final['PERCENT POSITIVE'], label='Percent Positive')
plt.title('Total Specimens, Total A, Total B, and Percent Positive')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))

