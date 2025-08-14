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

# Load training data for auto-fill functionality
try:
    training_data = pd.read_csv('supercars_train.csv')
    print("Training data loaded for auto-fill functionality!")
except Exception as e:
    print(f"Error loading training data: {e}")
    training_data = None

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

@app.route('/api/model-specs/<brand>/<model_name>')
def get_model_specs(brand, model_name):
    """Get typical specifications for a specific car model from training data"""
    if training_data is None:
        return jsonify({'error': 'Training data not available'}), 500
    
    try:
        # Filter data for the specific brand and model
        filtered_data = training_data[
            (training_data['brand'].str.lower() == brand.lower()) & 
            (training_data['model'].str.lower() == model_name.lower())
        ]
        
        if filtered_data.empty:
            return jsonify({'message': 'No data found for this model'})
        
        # Calculate typical values (median for numeric, mode for categorical)
        specs = {}
        
        # Numeric fields - use median
        numeric_fields = ['horsepower', 'torque', 'weight_kg', 'zero_to_60_s', 'top_speed_mph']
        for field in numeric_fields:
            if field in filtered_data.columns:
                specs[field] = float(filtered_data[field].median())
        
        # Categorical fields - use mode (most common)
        categorical_fields = ['engine_config', 'transmission', 'drivetrain', 'interior_material', 
                            'brake_type', 'tire_brand']
        for field in categorical_fields:
            if field in filtered_data.columns:
                mode_value = filtered_data[field].mode()
                if not mode_value.empty:
                    specs[field] = mode_value.iloc[0]
        
        # Boolean fields - use mode
        boolean_fields = ['carbon_fiber_body', 'aero_package', 'limited_edition']
        for field in boolean_fields:
            if field in filtered_data.columns:
                specs[field] = int(filtered_data[field].mode().iloc[0]) if not filtered_data[field].mode().empty else 0
        
        # Common values
        specs['num_doors'] = 2  # Most supercars are 2-door
        specs['mileage'] = 5000  # Typical low mileage
        specs['num_owners'] = 1  # Most are single owner
        specs['warranty_years'] = 0  # Default
        
        return jsonify({
            'specs': specs,
            'data_points': len(filtered_data),
            'message': f'Specifications based on {len(filtered_data)} similar vehicles'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<brand>')
def get_models_for_brand(brand):
    """Get available models for a specific brand from training data"""
    if training_data is None:
        return jsonify({'error': 'Training data not available'}), 500
    
    try:
        # Filter data for the specific brand
        brand_data = training_data[training_data['brand'].str.lower() == brand.lower()]
        
        if brand_data.empty:
            return jsonify([])
        
        # Get unique models and count of data points for each
        models = brand_data['model'].value_counts().to_dict()
        
        # Format as list of dictionaries with model name and count
        model_list = [{'name': model, 'count': count} for model, count in models.items()]
        
        return jsonify(model_list)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
