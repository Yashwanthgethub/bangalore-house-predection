from flask import Flask, jsonify, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
lr_clf = joblib.load('RidgeModel.pk.pkl')

# Dummy feature matrix X and column names (replace with your actual data)
# Assuming X contains features like 'sqft', 'bath', 'bhk', and location dummy variables
X = np.zeros((1, 4))  # Update the shape as per your data
X_columns = ['sqft', 'bath', 'bhk', 'location_dummy1', 'location_dummy2', ..., 'location_dummyn']

@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.json  # Assuming JSON input like {"location": "xyz", "sqft": 1000, "bath": 2, "bhk": 3}
    location = data['location']
    sqft = data['sqft']
    bath = data['bath']
    bhk = data['bhk']
    
    # Find the index of location in X_columns
    loc_index = X_columns.index(location)
    
    # Create the feature vector x
    x = np.zeros(len(X_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    
    # Predict the price
    predicted_price = lr_clf.predict([x])[0]
    
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
