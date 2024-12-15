# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/ASM Assignments/Classification Analysis/combined_data.csv'
data = pd.read_csv(file_path)

# Exploring the dataset
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())
print("\nSummary Statistics:")
print(data.describe())

# Step 1: Data Preprocessing
# Splitting data into features and labels
X = data['text']
y = data['label']

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorizing the text data
vectorizer = CountVectorizer(stop_words='english', max_features=2000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 2: Model Training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 3: Model Evaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Step 4: Visualization
# Histogram of spam vs not spam
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=data, palette='viridis')
plt.title('Distribution of Emails (Spam vs Not Spam)')
plt.xlabel('Label (0: Not Spam, 1: Spam)')
plt.ylabel('Count')
plt.show()