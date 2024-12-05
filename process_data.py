import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# ILINet: PERCENTAGE OF VISITS FOR INFLUENZA-LIKE-ILLNESS REPORTED BY SENTINEL PROVIDERS
# WHO_NREVSS_Clinical_Labs: Beginning for the 2015-16 season, reports from public health and clinical laboratories are presented separately in the weekly influenza update, FluView. Data from clinical laboratories include the weekly total number of specimens tested, the number of positive influenza test, and the percent positive by influenza type.
# WHO_NREVSS_Combined_prior_to_2015_16: Beginning for the 2015-16 season, reports from public health and clinical laboratories are presented separately in the weekly influenza update, FluView.  This data file includes only data prior to the 2015-16 influenza season, and will be presented with the public health and clinical labs combined.
# WHO_NREVSS_Public_Health_Labs: Beginning for the 2015-16 season, reports from public health and clinical laboratories are presented separately in the weekly influenza update, FluView.  Data presented from public health laboratories include the weekly total number of specimens tested, the number of positive influenza tests, and the number by influenza virus type, subtype, and influenza B lineage.

ILINet = pd.read_csv("dataset/ILINet.csv")
WHO_NREVSS_Clinical_Labs = pd.read_csv("dataset/WHO_NREVSS_Clinical_Labs.csv")
WHO_NREVSS_Combined_prior_to_2015_16 = pd.read_csv("dataset/WHO_NREVSS_Combined_prior_to_2015_16.csv")
WHO_NREVSS_Public_Health_Labs = pd.read_csv("dataset/WHO_NREVSS_Public_Health_Labs.csv")

# Drop constant columns
ILINet = ILINet.drop(columns=['REGION TYPE', 'REGION'])
WHO_NREVSS_Clinical_Labs = WHO_NREVSS_Clinical_Labs.drop(columns=['REGION TYPE', 'REGION'])
WHO_NREVSS_Combined_prior_to_2015_16 = WHO_NREVSS_Combined_prior_to_2015_16.drop(columns=['REGION TYPE', 'REGION'])
WHO_NREVSS_Public_Health_Labs = WHO_NREVSS_Public_Health_Labs.drop(columns=['REGION TYPE', 'REGION'])

#Normalize columns across all 
WHO_NREVSS_Combined_prior_to_2015_16_types = {'A': ['A (2009 H1N1)', 'A (H1)', 'A (H3)', 'A (Subtyping not Performed)', 'A (Unable to Subtype)', 'A (H5)'], 'B': ['B']}
WHO_NREVSS_Public_Health_Labs_types = {'A': ['A (2009 H1N1)', 'A (H3)', 'A (Subtyping not Performed)', 'H3N2v', 'A (H5)'], 'B': ['B', 'BVic', 'BYam']}

def filter_types(dataset, type_dict):
    dataset['TOTAL A'] = dataset[type_dict['A']].sum(axis=1)
    dataset['TOTAL B'] = dataset[type_dict['B']].sum(axis=1)
    dataset['PERCENT A'] = dataset['TOTAL A']/dataset['TOTAL SPECIMENS']
    dataset['PERCENT B'] = dataset['TOTAL B']/dataset['TOTAL SPECIMENS']
    
filter_types(WHO_NREVSS_Combined_prior_to_2015_16, WHO_NREVSS_Combined_prior_to_2015_16_types)
filter_types(WHO_NREVSS_Public_Health_Labs, WHO_NREVSS_Public_Health_Labs_types)
WHO_NREVSS_Public_Health_Labs['PERCENT POSITIVE'] = WHO_NREVSS_Public_Health_Labs[['TOTAL A', 'TOTAL B']].sum(axis=1)/WHO_NREVSS_Public_Health_Labs['TOTAL SPECIMENS']
WHO_NREVSS_Combined_prior_to_2015_16 = WHO_NREVSS_Combined_prior_to_2015_16[['YEAR', 'WEEK', 'TOTAL SPECIMENS', 'TOTAL A', 'TOTAL B', 'PERCENT POSITIVE', 'PERCENT A', 'PERCENT B']]
WHO_NREVSS_Public_Health_Labs=WHO_NREVSS_Public_Health_Labs[['YEAR', 'WEEK', 'TOTAL SPECIMENS', 'TOTAL A', 'TOTAL B', 'PERCENT POSITIVE', 'PERCENT A', 'PERCENT B']]
WHO_NREVSS_Combined_prior_to_2015_16.fillna(0, inplace=True)
WHO_NREVSS_Public_Health_Labs.fillna(0, inplace=True)

#Combine Data after 2015
concatination_after_2015 = pd.concat([WHO_NREVSS_Public_Health_Labs, WHO_NREVSS_Clinical_Labs])

WHO_NREVSS_Combined_after_2015 = (
    concatination_after_2015
    .groupby(['YEAR', 'WEEK'], as_index=False)
    .agg({
        'TOTAL SPECIMENS': 'sum',
        'TOTAL A': 'sum',
        'TOTAL B': 'sum'
    })
)
WHO_NREVSS_Combined_after_2015['PERCENT POSITIVE'] = (
    (WHO_NREVSS_Combined_after_2015['TOTAL A'] + WHO_NREVSS_Combined_after_2015['TOTAL B']) / 
    WHO_NREVSS_Combined_after_2015['TOTAL SPECIMENS']
)
WHO_NREVSS_Combined_after_2015['PERCENT A'] = (
    WHO_NREVSS_Combined_after_2015['TOTAL A'] / WHO_NREVSS_Combined_after_2015['TOTAL SPECIMENS']
)
WHO_NREVSS_Combined_after_2015['PERCENT B'] = (
    WHO_NREVSS_Combined_after_2015['TOTAL B'] / WHO_NREVSS_Combined_after_2015['TOTAL SPECIMENS']
)
WHO_NREVSS_Combined_after_2015.fillna(0, inplace=True)
#Combine WHO data
WHO_NVRES = pd.concat([WHO_NREVSS_Combined_prior_to_2015_16, WHO_NREVSS_Combined_after_2015], ignore_index=True)
WHO_NVRES = WHO_NVRES.sort_values(by=['YEAR', 'WEEK']).reset_index(drop=True)

#Combine all data
Combined_data = pd.merge(WHO_NVRES, ILINet, on=['YEAR', 'WEEK'], how='inner')

# Impute missing values
Combined_data = Combined_data.replace('X', np.nan)
most_frequent_columns = ['AGE 25-49', 'AGE 50-64', 'AGE 25-64', '% WEIGHTED ILI']
zero_columns = ['YEAR', 'WEEK', 'TOTAL SPECIMENS', 'TOTAL A', 'TOTAL B', 'PERCENT POSITIVE', 'PERCENT A', 'PERCENT B', '% WEIGHTED ILI', '%UNWEIGHTED ILI', 'ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS']

imputer_most_frequent = SimpleImputer(strategy='most_frequent')
imputer_zero = SimpleImputer(strategy='constant', fill_value=0)

Combined_data[most_frequent_columns] = imputer_most_frequent.fit_transform(Combined_data[most_frequent_columns])
Combined_data[zero_columns] = imputer_zero.fit_transform(Combined_data[zero_columns])

# Check if all missing values are filled
print(Combined_data.isnull().sum())
print(Combined_data.isna().sum())

# Convert the year and week columns to datetime and set it as the index
Combined_data['date'] = pd.to_datetime(Combined_data[['YEAR', 'WEEK']].astype(str).agg('-'.join, axis=1) + '-1', format='%Y-%U-%w')
merged_df = Combined_data.set_index('date')
merged_df = merged_df.drop(columns=['YEAR', 'WEEK'])

merged_df.rename(columns={"% WEIGHTED ILI": "OT"}, inplace=True)

# Save the DataFrame to a new CSV file
merged_df.to_csv('dataset/flu.csv', index=True)
