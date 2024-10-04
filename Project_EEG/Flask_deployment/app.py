from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    ann_model = pickle.load(file)

# Load the scaler for input data
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    data = {
        'Fp1': float(request.form['Fp1']),
        'Fp2': float(request.form['Fp2']),
        'F7': float(request.form['F7']),
        'F8': float(request.form['F8']),
        'T3': float(request.form['T3']),
        'T4': float(request.form['T4']),
        'P3': float(request.form['P3']),
        'P4': float(request.form['P4']),
        'O1': float(request.form['O1']),
        'O2': float(request.form['O2']),
        'Delta Power': float(request.form['Delta_Power']),
        'Theta Power': float(request.form['Theta_Power']),
        'Alpha Power': float(request.form['Alpha_Power']),
        'Beta Power': float(request.form['Beta_Power']),
        'Gamma Power': float(request.form['Gamma_Power']),
        'Signal Quality_Fair': 1 if request.form['Signal_Quality'] == 'Fair' else 0,
        'Signal Quality_Good': 1 if request.form['Signal_Quality'] == 'Good' else 0,
        'Signal Quality_Poor': 1 if request.form['Signal_Quality'] == 'Poor' else 0,
        'Task Difficulty_Low': 1 if request.form['Task_Difficulty'] == 'Low' else 0,
        'Task Difficulty_Medium': 1 if request.form['Task_Difficulty'] == 'Medium' else 0,
        'Mental Fatigue_Low': 1 if request.form['Mental_Fatigue'] == 'Low' else 0,
        'Mental Fatigue_Medium': 1 if request.form['Mental_Fatigue'] == 'Medium' else 0
    }

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([data])

    # Preprocess the input data
    input_data_scaled = scaler.transform(input_df)

    # Make prediction using the ANN model
    prediction = ann_model.predict(input_data_scaled)

    # Convert one-hot encoded prediction to a single class label
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map the prediction to attention level
    attention_level_mapping = {0: "High", 1: "Low", 2: "Medium"}
    attention_level = attention_level_mapping.get(predicted_class, "Unknown")

    # Redirect to results page with attention level
    return redirect(url_for('result', level=attention_level))

# Define the results route
@app.route('/result')
def result():
    level = request.args.get('level', 'Unknown')
    return render_template('result.html', attention_level=level)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
