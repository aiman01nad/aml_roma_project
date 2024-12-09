import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

flu_final = pd.read_csv('dataset/flu_enhanced.csv')
flu = pd.read_csv('dataset/flu.csv')

flu_final['date'] = pd.to_datetime(flu_final['date'])

corr_matrix = flu_final.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()

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
plt.plot(flu_final['date'], flu_final['PERCENT A'], label='PERCENT A')
plt.plot(flu_final['date'], flu_final['PERCENT B'], label='PERCENT B')
plt.plot(flu_final['date'], flu_final['PERCENT POSITIVE'], label='Percent Positive')
plt.title('Percentage of Positive Flu Tests')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.show()