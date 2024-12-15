# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Load the dataset
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/ASM Assignments/Survival Analysis/lung_cancer_data.csv'
data = pd.read_csv(file_path)

# Exploring the dataset
print("Dataset Info: ")
print(data.info())
print("\nFirst 5 rows: ")
print(data.head())
print("\nSummary Statistics: ")
print(data.describe())

# Filter relevant columns
relevant_columns = ['Treatment', 'Survival_Months']
data = data[relevant_columns]

# Drop rows with missing values
data = data.dropna()

# Group data by treatment and calculate mean survival months
mean_survival = data.groupby('Treatment')['Survival_Months'].mean().sort_values(ascending=False)

# Display mean survival
print("Mean Survival Months by Treatment:")
print(mean_survival)

# Boxplot to visualize survival months by treatment
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='Treatment', y='Survival_Months', palette='Set2')
plt.title('Survival Months by Treatment')
plt.xlabel('Treatment')
plt.ylabel('Survival Months')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Perform ANOVA to check for statistical significance
groups = [data[data['Treatment'] == treatment]['Survival_Months'] for treatment in data['Treatment'].unique()]
anova_result = f_oneway(*groups)

print("\nANOVA Test Result:")
print(f"F-statistic: {anova_result.statistic:.2f}, p-value: {anova_result.pvalue:.4f}")

# Interpretation of ANOVA result
if anova_result.pvalue < 0.05:
    print("The differences in survival months between treatments are statistically significant.")
else:
    print("No statistically significant differences in survival months between treatments.")