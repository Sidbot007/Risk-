Heart Disease Risk Prediction
Overview
This project is a web application designed to predict the risk of heart disease based on user input. It integrates a machine learning model with a Flask backend to provide users with an easy-to-use interface for submitting their health data and receiving risk predictions.

Features
User Input Form: Collects user data such as age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate, exercise-induced angina, ST depression, slope of peak exercise ST segment, number of major vessels, and thalassemia.
Machine Learning Integration: Uses a pre-trained Gradient Boosting Classifier model to predict the risk of heart disease.
Risk Percentage Calculation: Converts model output into a user-friendly risk percentage.
Error Handling: Provides meaningful feedback to users in case of errors during prediction.
JSON Response: Returns risk prediction in JSON format for frontend consumption.



## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository**:
    ```bash
     git clone https://github.com/Sidbot007/Risk.git
     cd Risk
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

Ensure you have the following datasets in the root directory:
- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data`

## Training the Model

1. **Run the training script**:
    ```bash
    python train_model.py
    ```
   This script will:
   - Load and combine datasets
   - Preprocess the data (handling missing values, one-hot encoding)
   - Scale the features
   - Handle imbalanced data using SMOTE
   - Split the data into training and testing sets
   - Perform a grid search to find the best hyperparameters for the Gradient Boosting Classifier
   - Save the best model and scaler using `joblib`

## Running the Application

1. **Start the Flask server**:
    ```bash
    flask run
    ```

2. **Access the application**:
    Open your web browser and go to `http://127.0.0.1:5000/`.

## Usage

Fill in the form on the web page with the required input parameters and submit it to get the heart disease risk prediction.

## Files

- `app.py`: The main Flask application file.
- `train_model.py`: Script for training and saving the machine learning model.
- `requirements.txt`: Contains the list of required Python packages.
- `templates/index.html`: The HTML form for user input.

## Future Enhancements

- Improve the user interface for better user experience.
- Add more features to the prediction model.
- Implement user authentication.

## Contributing

Feel free to fork this repository and submit pull requests.

## License

This project is licensed under the MIT License.




















