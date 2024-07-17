from flask import Flask, jsonify, render_template, request
import numpy as np
import joblib
import os
import pandas as pd


app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load the model, scaler, and feature names
try:
    model = joblib.load('best_heart_disease_gb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError as e:
    print(f"Error loading model, scaler, or feature names: {e}")
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
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'age': [float(data['age'])],
            'sex': [float(data['sex'])],
            'cp': [float(data['cp'])],
            'trestbps': [float(data['trestbps'])],
            'chol': [float(data['chol'])],
            'fbs': [float(data['fbs'])],
            'restecg': [float(data['restecg'])],
            'thalach': [float(data['thalach'])],
            'exang': [float(data['exang'])],
            'oldpeak': [float(data['oldpeak'])],
            'slope': [float(data['slope'])],
            'ca': [float(data['ca'])],
            'thal': [float(data['thal'])]
        })

        # One-hot encode the input data to match training features
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

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

# Route to display the classification report
@app.route('/report')
def report():
    if os.path.exists('classification_report.txt'):
        with open('classification_report.txt', 'r') as f:
            report_content = f.read()
        return render_template('report.html', report=report_content)
    else:
        return render_template('error.html'), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
