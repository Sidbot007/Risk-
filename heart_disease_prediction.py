import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
file_path = "processed.cleveland.data"  # Ensure this file is in your project folder
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
data = pd.read_csv(file_path, names=column_names)

# Replace '?' with NaN and convert to appropriate dtype
data = data.replace('?', np.nan).astype(float)

# Fill missing values with median
data = data.fillna(data.median())

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'])

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save the column names after one-hot encoding
joblib.dump(X.columns, 'columns.pkl')

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model trained and saved successfully.")
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
