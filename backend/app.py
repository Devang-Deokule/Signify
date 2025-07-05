import tensorflow as tf
import cv2
import numpy as np
import pyttsx3

# Constants
IMG_SIZE = 64
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# Load trained model
model = tf.keras.models.load_model("asl_model.h5")
print("✅ Model loaded.")

# Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 130)

# Start webcam
cap = cv2.VideoCapture(0)
print("🎥 Webcam started — press 'q' to quit.")

sentence = ""
last_prediction = ""
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Region of interest
    x1, y1, size = 100, 100, 200
    x2, y2 = x1 + size, y1 + size
    roi = frame[y1:y2, x1:x2]

    # Preprocess
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_normalized = roi_resized / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    # Predict every 20 frames (avoid flicker)
    if frame_count % 20 == 0:
        prediction = model.predict(roi_input)
        pred_idx = np.argmax(prediction)
        pred_class = CLASSES[pred_idx]

        if pred_class != last_prediction:
            last_prediction = pred_class
            print(f"🧠 Predicted: {pred_class}")

            if pred_class == "space":
                sentence += " "
            elif pred_class == "del":
                sentence = sentence[:-1]
            elif pred_class == "nothing":
                pass
            else:
                sentence += pred_class

            # Speak the prediction
            if pred_class not in ["del", "nothing"]:
                engine.say(pred_class)
                engine.runAndWait()

    frame_count += 1

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display current prediction + sentence
    cv2.putText(frame, f"Prediction: {last_prediction}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (50, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
