import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_dict = joblib.load("model.p")
model = model_dict["model"]

classes = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'M',
    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y',
    24: 'Z', 25: ' '
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        landmarks = data.get("landmarks")

        print("\n[INFO] Received landmarks:")
        print(f"Length of input: {len(landmarks)}")
        print(f"Data sample: {landmarks[:6]}")

        if not landmarks or len(landmarks) != 42:
            return jsonify({"prediction": "Invalid"})

        prediction = model.predict([landmarks])[0]
        print("[PREDICTION]", prediction)

        result = classes.get(prediction, "Unknown")
        return jsonify({"prediction": result})

    except Exception as e:
        print("[EXCEPTION]", str(e))
        return jsonify({"prediction": "Error"})

if __name__ == "__main__":
    app.run(debug=True)