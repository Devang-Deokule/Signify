
            

import os
import pickle
import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to the data directory (adjust if needed)
DATA_DIR = r'C:\Users\viraj\OneDrive\Desktop\asl-detection-app\data'

data = []
labels = []

# Loop through each class folder like 0, 1, 2, ...
for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.isdir(class_path):
        continue  # skip files like A.jpg, B.jpg, etc.

    # Loop through all images in the class folder
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Couldn't load image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []

            hand_landmarks = results.multi_hand_landmarks[0]

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(int(class_name))  # Store label as integer

# Close MediaPipe
hands.close()

# Save features and labels to pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"[DONE] Saved {len(data)} samples to data.pickle")
