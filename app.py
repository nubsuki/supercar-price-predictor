from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained model and feature info
try:
    model = joblib.load('supercar_price_model.pkl')
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
    print("Model loaded successfully!")
    print(f"Best model: {model_info['best_model']}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_info = None

@app.route('/')
def home():
    if model is None:
        return " Model not loaded"
    
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert numeric fields
        numeric_fields = ['year', 'horsepower', 'torque', 'weight_kg', 'zero_to_60_s', 
                         'top_speed_mph', 'num_doors', 'mileage', 'num_owners', 
                         'warranty_years', 'damage_cost']
        
        for field in numeric_fields:
            if field in data and data[field]:
                try:
                    data[field] = float(data[field])
                except ValueError:
                    data[field] = 0
        
        # Convert boolean fields
        boolean_fields = ['carbon_fiber_body', 'aero_package', 'limited_edition', 
                         'has_warranty', 'non_original_parts', 'damage']
        
        for field in boolean_fields:
            data[field] = 1 if data.get(field) == 'on' else 0
        
        # Handle last_service_date
        if 'last_service_date' in data and data['last_service_date']:
            service_date = pd.to_datetime(data['last_service_date'])
            latest_date = datetime.now()
            data['days_since_service'] = (latest_date - service_date).days
        else:
            data['days_since_service'] = 365  # Default to 1 year
        
        # Remove last_service_date as it's converted to days_since_service
        if 'last_service_date' in data:
            del data['last_service_date']
        
        # Handle damage_type
        if data.get('damage') == 0:
            data['damage_type'] = 'none'
        
        # Create DataFrame for prediction
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return jsonify({
            'prediction': f"${prediction:,.2f}",
            'prediction_value': prediction
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/brands')
def get_brands():
    # Common supercar brands from the dataset
    brands = ['McLaren', 'Ferrari', 'Lamborghini', 'Porsche', 'Aston Martin', 
              'Bugatti', 'Koenigsegg', 'Pagani', 'Maserati', 'Lotus', 'Bentley']
    return jsonify(brands)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
