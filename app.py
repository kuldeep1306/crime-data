from flask import Flask, request, jsonify, render_template

import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model + encoder
model, le = pickle.load(open("crime_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    state = data["state"]
    year = int(data["year"])

    # Encode state
    state_encoded = le.transform([state])[0]

    # Prepare input
    input_data = np.array([[state_encoded, year]])

    prediction = model.predict(input_data)

    result = {
        "state": state,
        "year": year,
        "total_predicted": int(prediction[0]),
        "latitude": 28.7041,   # You can map dynamically later
        "longitude": 77.1025,
        "color": "red"
    }

    return jsonify(result)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
