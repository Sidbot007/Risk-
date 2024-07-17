import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Define file paths for all datasets
file_paths = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data"
]

# Column names for the dataset
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load and combine all datasets
dataframes = []
for file_path in file_paths:
    df = pd.read_csv(file_path, names=column_names)
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

# Replace '?' with NaN and convert to appropriate dtype
data = data.replace('?', np.nan).astype(float)

# Fill missing values with median
data = data.fillna(data.median())

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'])

# Save the column names after one-hot encoding
feature_names = data.columns.drop('target').tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalanced data with SMOTE
oversample = SMOTE(random_state=42)
X_resampled, y_resampled = oversample.fit_resample(X_scaled, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define the model
model = GradientBoostingClassifier(random_state=42)

# Define hyperparameters grid (reduced for faster training)
param_grid = {
    'n_estimators': [100, 200],  # Reduced number of estimators
    'learning_rate': [0.1],      # Focus on a single learning rate
    'max_depth': [3, 4],         # Reduce depth range
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 1.0]      # Reduce subsample range
}

# Perform Grid Search with reduced cross-validation folds
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, 'best_heart_disease_gb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save the column names after one-hot encoding
joblib.dump(X.columns, 'columns.pkl')

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save the classification report
with open('classification_report.txt', 'w') as f:
    f.write(report)

print("Best Gradient Boosting model trained and saved successfully.")
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
