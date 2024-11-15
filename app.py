from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, auth, firestore
import pickle

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Load the pickled models
with open('lr100_model.pkl', 'rb') as model_file:
    linear_model = pickle.load(model_file)
with open('rfr100_model.pkl', 'rb') as model_file:
    rfr_model = pickle.load(model_file)
with open('crop_recommendation_model.pkl', 'rb') as model_file:
    rec_model = pickle.load(model_file)

@app.route('/test', methods=['GET'])
def test():
    # Dummy data for testing
    dummy_data = {
        "Rainfall": 100,
        "Temperature": 25,
        "Nitrogen": 50,
        "Phosphorus": 20,
        "Potassium": 30,
        "pH": 6.5,
        "Humidity": 60,
        "plot_size": 2,
        "crop": "wheat",  # Example crop for testing
        "price": 10
    }

    # Crop categories
    crop_types = {
        'cereals': ['rice', 'maize'],
        'legumes': ['chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil'],
        'fruits': ['pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut'],
        'cash_crops': ['cotton', 'jute', 'coffee']
    }

    # Function to determine crop type
    def get_crop_type(crop):
        for category, crops in crop_types.items():
            if crop.lower() in crops:
                return category
        return None

    try:
        # Extract features for prediction
        pred_features = [dummy_data[key] for key in ['Rainfall', 'Temperature', 'Nitrogen', 'Phosphorus', 'Potassium']]
        rec_features = [dummy_data[key] for key in ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']]
        
        # Make predictions
        linear_prediction = linear_model.predict([pred_features])[0]
        rfr_prediction = rfr_model.predict([pred_features])[0]
        rec_prediction = rec_model.predict([rec_features])[0]  # Recommended crop

        # Determine the crop type of both
        actual_crop = dummy_data['crop']
        actual_crop_type = get_crop_type(actual_crop)
        rec_crop_type = get_crop_type(rec_prediction)

        # Determine suitability factor
        if actual_crop.lower() == rec_prediction.lower():
            suitability_factor = 1  # Exact match
        elif actual_crop_type == rec_crop_type:
            suitability_factor = 0.9  # Same type
        else:
            suitability_factor = 0.7  # Different type

        # Adjust revenue based on suitability factor
        plot_size = dummy_data['plot_size']
        price_per_kg = dummy_data['price']
        estimated_revenue_linear = linear_prediction * plot_size * price_per_kg * suitability_factor
        estimated_revenue_rfr = rfr_prediction * plot_size * price_per_kg * suitability_factor

        # Prepare the response
        response = {
            'linear_prediction': linear_prediction,
            'rfr_prediction': rfr_prediction,
            'estimated_revenue_linear': estimated_revenue_linear,
            'estimated_revenue_rfr': estimated_revenue_rfr,
            'rec_prediction': rec_prediction,
            'suitability_factor': suitability_factor
        }
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the request data
        data = request.get_json()

        # Check if all required keys are present
        required_keys = ['Rainfall', 'Temperature', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Humidity', 'plot_size', 'crop', 'price']
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Missing one or more required fields'}), 400

        # Extract features for prediction
        features = [data[key] for key in ['Rainfall', 'Temperature', 'Nitrogen', 'Phosphorus', 'Potassium']]
        
        # Make predictions
        linear_prediction = linear_model.predict([features])[0]
        rfr_prediction = rfr_model.predict([features])[0]

        # Calculate revenue based on predictions and plot size/price
        plot_size = data['plot_size']
        price_per_kg = data['price']
        estimated_revenue_linear = linear_prediction * plot_size * price_per_kg
        estimated_revenue_rfr = rfr_prediction * plot_size * price_per_kg

        # Prepare data to store in Firestore
        prediction_data = {
            'linear_prediction': linear_prediction,
            'rfr_prediction': rfr_prediction,
            'estimated_revenue_linear': estimated_revenue_linear,
            'estimated_revenue_rfr': estimated_revenue_rfr,
            'crop': data['crop'],
            'timestamp': firestore.SERVER_TIMESTAMP
        }

        # Update Firestore
        db.collection('crop_predictions').add(prediction_data)

        # Return the response
        response = {
            'linear_prediction': linear_prediction,
            'rfr_prediction': rfr_prediction,
            'estimated_revenue_linear': estimated_revenue_linear,
            'estimated_revenue_rfr': estimated_revenue_rfr
        }
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'response': 'pong'})

# Delete Firebase user
@app.route('/delete_user', methods=['POST'])
def delete_user():
    uid = request.json.get('uid')

    if not uid:
        return jsonify({'error': 'Missing user ID'}), 400

    try:
        auth.delete_user(uid)
        return jsonify({'message': 'User deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)