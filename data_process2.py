import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# ILINet: PERCENTAGE OF VISITS FOR INFLUENZA-LIKE-ILLNESS REPORTED BY SENTINEL PROVIDERS
# WHO_NREVSS_Clinical_Labs: Beginning for the 2015-16 season, reports from public health and clinical laboratories are presented separately in the weekly influenza update, FluView. Data from clinical laboratories include the weekly total number of specimens tested, the number of positive influenza test, and the percent positive by influenza type.
# WHO_NREVSS_Combined_prior_to_2015_16: Beginning for the 2015-16 season, reports from public health and clinical laboratories are presented separately in the weekly influenza update, FluView.  This data file includes only data prior to the 2015-16 influenza season, and will be presented with the public health and clinical labs combined.
# WHO_NREVSS_Public_Health_Labs: Beginning for the 2015-16 season, reports from public health and clinical laboratories are presented separately in the weekly influenza update, FluView.  Data presented from public health laboratories include the weekly total number of specimens tested, the number of positive influenza tests, and the number by influenza virus type, subtype, and influenza B lineage.

ILINet = pd.read_csv("dataset/ILINet.csv")
WHO_clinical = pd.read_csv("dataset/WHO_NREVSS_Clinical_Labs.csv")
WHO_combined_pre_2015 = pd.read_csv("dataset/WHO_NREVSS_Combined_prior_to_2015_16.csv")
WHO_public = pd.read_csv("dataset/WHO_NREVSS_Public_Health_Labs.csv")

# Drop constant columns
for dataset in [ILINet, WHO_clinical, WHO_combined_pre_2015, WHO_public]:
    dataset.drop(columns=['REGION TYPE', 'REGION'], inplace=True)

# ILINet: Combine the "AGE 25-49" and "AGE 50-64" groups into "AGE 25-64" across the whole time period
age_columns = ['AGE 25-64', 'AGE 25-49', 'AGE 50-64']
for column in age_columns:
    ILINet[column] = pd.to_numeric(ILINet[column], errors='coerce')

ILINet['AGE 25-64'] = ILINet.apply(
    lambda row: row['AGE 25-64'] if not pd.isnull(row['AGE 25-64']) 
    else row.get('AGE 25-49', 0) + row.get('AGE 50-64', 0),
    axis=1
)

ILINet = ILINet.drop(columns=['AGE 25-49', 'AGE 50-64'], errors='ignore')
ILINet['AGE 25-64'] = ILINet['AGE 25-64'].astype(int)

ILINet['% WEIGHTED ILI'] = pd.to_numeric(ILINet['% WEIGHTED ILI'], errors='coerce') # Replace 'X' with NaN

# WHO Combined prior to 2015-16: Combine all "A" types into "TOTAL A"
def combine_subtypes(data, type, new_column):
    if type == 'A':
        subtypes = [col for col in data.columns if 'A (' in col]
        subtypes.append('H3N2v')
    elif type == 'B':
        subtypes = [col for col in data.columns if 'B' in col]
    else:
        return
    data[subtypes] = data[subtypes].fillna(0)
    data[new_column] = data[subtypes].sum(axis=1)
    data.drop(columns=subtypes, inplace=True)

combine_subtypes(WHO_combined_pre_2015, 'A', 'TOTAL A')

WHO_combined_pre_2015.rename(columns={'B': 'TOTAL B'}, inplace=True)   # Rename 'B' column to 'TOTAL B'
WHO_combined_pre_2015.drop(columns=['PERCENT POSITIVE'], inplace=True) # Drop PERCENT POSITIVE column

# WHO Public health labs: Combine all "A" types into "TOTAL A" and all "B" types into "TOTAL B"
combine_subtypes(WHO_public, 'A', 'TOTAL A')
combine_subtypes(WHO_public, 'B', 'TOTAL B')

# Merge WHO public health labs and clinical labs
WHO_combined_after_2015 = pd.DataFrame(columns=['YEAR', 'WEEK', 'TOTAL SPECIMENS', 'TOTAL A', 'TOTAL B'])
WHO_combined_after_2015['YEAR'] = WHO_public['YEAR']
WHO_combined_after_2015['WEEK'] = WHO_public['WEEK']
WHO_combined_after_2015['TOTAL SPECIMENS'] = WHO_public['TOTAL SPECIMENS'] + WHO_clinical['TOTAL SPECIMENS']
WHO_combined_after_2015['TOTAL A'] = WHO_public['TOTAL A'] + WHO_clinical['TOTAL A']
WHO_combined_after_2015['TOTAL B'] = WHO_public['TOTAL B'] + WHO_clinical['TOTAL B']

# Combine all WHO datasets
who_combined = pd.concat([WHO_combined_pre_2015, WHO_combined_after_2015], ignore_index=True)
who_combined.to_csv("dataset/WHO_Combined.csv", index=False)

# Merge with ILINet
merged_data = pd.merge(ILINet, who_combined, on=['YEAR', 'WEEK'], how='outer')
merged_data.to_csv('dataset/merged_data.csv', index=False)

# Impute missing values 
merged_data.replace(0.0, np.nan, inplace=True)
#imputer = SimpleImputer(strategy='mean')
imputer = KNNImputer(n_neighbors=5)
merged_data[:] = imputer.fit_transform(merged_data)

# Scale the data
exclude_columns = ['YEAR', 'WEEK', '% WEIGHTED ILI']
columns_to_scale = [col for col in merged_data.columns if col not in exclude_columns]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data[columns_to_scale])
scaled_data = pd.DataFrame(scaled_data, columns=columns_to_scale, index=merged_data.index)
merged_data = pd.concat([merged_data[exclude_columns], scaled_data], axis=1)

# Convert the year and week columns to datetime and set it as the index
merged_data['date'] = pd.to_datetime(merged_data[['YEAR', 'WEEK']].astype(str).agg('-'.join, axis=1) + '-1', format='%Y-%U-%w')
merged_data = merged_data.set_index('date')
merged_data = merged_data.drop(columns=['YEAR', 'WEEK'])

# Set target variable
merged_data.rename(columns={"% WEIGHTED ILI": "OT"}, inplace=True)

# Save the DataFrame to a new CSV file
merged_data.to_csv('dataset/flu.csv', index=True)

# Plot the data
plt.figure(figsize=(12, 6))
#plt.plot(merged_data.index, merged_data['OT'], label='% Weighted ILI', color='blue')
plt.plot(merged_data.index, merged_data['TOTAL A'], label='Total A', color='red', alpha=0.7)
plt.plot(merged_data.index, merged_data['TOTAL B'], label='Total B', color='green', alpha=0.7)
#plt.plot(merged_data.index, merged_data['TOTAL SPECIMENS'], label='Total Specimens', color='orange', alpha=0.7)
#plt.plot(merged_data.index, merged_data['TOTAL PATIENTS'], label='Total Patients', color='purple', alpha=0.7)
plt.title('Time Series of Key Metrics')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.show()

""" plt.figure(figsize=(10, 8))
sns.heatmap(merged_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show() """


