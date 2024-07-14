from flask import Flask, jsonify, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    exit(1)

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Ensure all form data is converted to the correct type
        input_data = np.array([[
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]])
        
        # Transform the input data using the scaler
        scaled_input = scaler.transform(input_data)
        
        # Predict the risk using the model
        prediction = model.predict_proba(scaled_input)
        risk_percentage = prediction[0][1] * 100
        
        # Render result.html with the risk_percentage value
        return render_template('result.html', risk_percentage=risk_percentage)
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return render_template('error.html'), 500
