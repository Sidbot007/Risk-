# train_model.py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import numpy as np

# Load your dataset with proper header specification and handle missing values
data = pd.read_csv('processed.cleveland.data', header=None, names=[
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
])

# Replace '?' with NaN and convert numeric columns to float
data = data.replace('?', np.nan)
data = data.astype(float)

# Impute missing values with mean
data = data.fillna(data.mean())

# Select the required features
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = data[features]
y = data['target']

# Fit the scaler and model
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Save the scaler and model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'heart_disease_model.pkl')

print("Model training and saving completed successfully.")

