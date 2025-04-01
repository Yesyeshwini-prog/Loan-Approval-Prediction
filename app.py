from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature names for the input form
feature_columns = [
    'no_of_dependents', 'education', 'self_employed', 'income_annum', 
    'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 
    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

# Home route to display the input form
@app.route('/')
def home():
    return render_template('index.html')

# About Dataset route
@app.route('/about-dataset')
def about_dataset():
    return render_template('about_dataset.html')

# Model Performance route
@app.route('/model-performance')
def model_performance():
    return render_template('model_performance.html')

# Prediction route to handle the form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        no_of_dependents = int(request.form['no_of_dependents'])
        education = 1 if request.form['education'] == 'Graduate' else 0
        self_employed = 1 if request.form['self_employed'] == 'Yes' else 0
        income_annum = float(request.form['income_annum'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = int(request.form['loan_term'])
        cibil_score = float(request.form['cibil_score'])
        residential_assets_value = float(request.form['residential_assets_value'])
        commercial_assets_value = float(request.form['commercial_assets_value'])
        luxury_assets_value = float(request.form['luxury_assets_value'])
        bank_asset_value = float(request.form['bank_asset_value'])

        # Create a DataFrame for input data
        input_data = pd.DataFrame([[no_of_dependents, education, self_employed, income_annum, loan_amount,
                                    loan_term, cibil_score, residential_assets_value, commercial_assets_value, 
                                    luxury_assets_value, bank_asset_value]],
                                  columns=feature_columns)
        
        # Normalize the numerical columns using the same scaler
        numerical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                          'residential_assets_value', 'commercial_assets_value',
                          'luxury_assets_value', 'bank_asset_value']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # Make prediction
        prediction = model.predict(input_data)

        # Display the result
        result = 'Approved' if prediction[0] == 1 else 'Rejected'

        return render_template('index.html', prediction_text=f'Loan {result}')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
