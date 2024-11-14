from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, auth

import pickle

app = Flask(__name__)
# Initialize Firebase Admin SDK
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

# Load the pickled model
with open('lr100_model.pkl', 'rb') as model_file:
    linear_model = pickle.load(model_file)
with open('rfr100_model.pkl', 'rb') as model_file:
    rfr_model = pickle.load(model_file)

@app.route('/test', )
def test():
    return 'API is working'

@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    data = request.get_json()
    
    # Extract features from the request data
    features = [data['Rainfall'], data['Temperature'], data['Nitrogen'], data['Phosphorus'], data['Potassium']]  # Adjust based on your model's features
    
    # Make prediction
    linear_prediction = linear_model.predict([features])
    
    rfr_prediction = rfr_model.predict([features])

    # Combine the predictions as response
    response = {'linear_prediction': linear_prediction[0], 'rfr_prediction': rfr_prediction[0]}

    # Return the prediction as a JSON response
    return jsonify({'response': response})

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'response': 'pong'})

# delete firebase user
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