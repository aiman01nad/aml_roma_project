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

# Renaming the total specimens column
WHO_NREVSS_Clinical_Labs.rename(columns={"TOTAL SPECIMENS": "total_specimens_clinical_labs"}, inplace=True)
WHO_NREVSS_Combined_prior_to_2015_16.rename(columns={"TOTAL SPECIMENS": "total_specimens_combined"}, inplace=True)
WHO_NREVSS_Public_Health_Labs.rename(columns={"TOTAL SPECIMENS": "total_specimens_public_health"}, inplace=True)

# Drop constant columns
ILINet = ILINet.drop(columns=['REGION TYPE', 'REGION'])
WHO_NREVSS_Clinical_Labs = WHO_NREVSS_Clinical_Labs.drop(columns=['REGION TYPE', 'REGION'])
WHO_NREVSS_Combined_prior_to_2015_16 = WHO_NREVSS_Combined_prior_to_2015_16.drop(columns=['REGION TYPE', 'REGION'])
WHO_NREVSS_Public_Health_Labs = WHO_NREVSS_Public_Health_Labs.drop(columns=['REGION TYPE', 'REGION'])

# Merge the datasets
pre_2015_16_df = pd.merge(ILINet, WHO_NREVSS_Combined_prior_to_2015_16, on=["YEAR", "WEEK"], how="outer")
post_2015_16_df = pd.merge(WHO_NREVSS_Clinical_Labs, WHO_NREVSS_Public_Health_Labs, on=["YEAR", "WEEK"], how="outer")
merged_df = pd.merge(pre_2015_16_df, post_2015_16_df, on=["YEAR", "WEEK"], how="outer")

merged_df.to_csv('dataset/merged_fludata.csv', index=False)

# Impute missing values
most_frequent_columns = ['AGE 25-49', 'AGE 50-64']
zero_columns = ['total_specimens_combined', 'PERCENT POSITIVE_x', 'A (2009 H1N1)_x', 'A (H1)', 'A (H3)_x', 'A (Subtyping not Performed)_x', 'A (Unable to Subtype)', 'B_x', 'H3N2v_x', 'A (H5)_x' ,'total_specimens_clinical_labs', 'TOTAL A', 'TOTAL B', 'PERCENT POSITIVE_y' ,'PERCENT A', 'PERCENT B','total_specimens_public_health', 'A (2009 H1N1)_y', 'A (H3)_y', 'A (Subtyping not Performed)_y', 'B_y' ,'BVic', 'BYam', 'H3N2v_y', 'A (H5)_y']

imputer_most_frequent = SimpleImputer(strategy='most_frequent')
imputer_zero = SimpleImputer(strategy='constant', fill_value=0)

merged_df[most_frequent_columns] = imputer_most_frequent.fit_transform(merged_df[most_frequent_columns])
merged_df[zero_columns] = imputer_zero.fit_transform(merged_df[zero_columns])

# Check if all missing values are filled
print(merged_df.isnull().sum())

# Save the DataFrame to a new CSV file
merged_df.to_csv('dataset/processed_fludata.csv', index=False)





