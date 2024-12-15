# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
from google.colab import drive
file_path = "drive/My Drive/ASM Assignments/Predicting House Prices/BostonHousing.csv"
data = pd.read_csv(file_path)

# Step 2: Explore the dataset
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())
print("\nSummary statistics:")
print(data.describe())

# Step 3: Handle missing values
data['rm'].fillna(data['rm'].median(), inplace=True)

# Step 4: Select features and target
# Choosing relevant features based on domain knowledge
features = ['crim', 'rm', 'lstat', 'ptratio', 'nox', 'tax', 'dis']
target = 'medv'

X = data[features]
y = data[target]

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 9: Visualize predictions vs actual values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Prices (medv)")
plt.ylabel("Predicted Prices (medv)")
plt.title("Actual vs Predicted House Prices")
plt.show()