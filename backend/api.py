from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model("asl_model.h5")
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']
IMG_SIZE = 64

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    img_bytes = base64.b64decode(data.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions)
    confidence = float(np.max(predictions))  # highest probability
    pred_class = CLASSES[pred_idx]

    return jsonify({'prediction': pred_class, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
