# rental_app.py

from flask import Flask, request, render_template
import pandas as pd
import joblib

# Load the trained model and encoder
model = joblib.load('rental_price_model.pkl')
encoder = joblib.load('region_encoder.pkl')

app = Flask(__name__)

# Home page with form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from the form
        region = request.form['region']
        no_of_bedrooms = int(request.form['bedrooms'])

        # Create a DataFrame for the input
        new_data = pd.DataFrame({'No of Bedroom': [no_of_bedrooms], 'REGION': [region]})

        # Encode the region using the same encoder used in training
        new_data_encoded = pd.concat(
            [new_data[['No of Bedroom']],
             pd.DataFrame(encoder.transform(new_data[['REGION']]), columns=encoder.get_feature_names_out(['REGION']))],
            axis=1
        )

        # Make the prediction
        prediction = model.predict(new_data_encoded)
        predicted_rent = round(prediction[0], 2)

        # Render the result on the page
        return render_template('index.html', prediction_text=f'Predicted Monthly Rent: ${predicted_rent}')

if __name__ == "__main__":
    app.run(debug=True)

