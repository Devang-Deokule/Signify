import os
import cv2

DATA_DIR = r'C:\Users\viraj\OneDrive\Desktop\asl-detection-app\data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

# Try default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Couldn't access webcam at index 0. Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[FATAL] No webcam found.")
        exit()

for j in range(number_of_classes):
    class_path = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_path, exist_ok=True)

    print('Collecting data for class {}'.format(j))

    # Wait for user confirmation to start
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to grab frame.")
            continue

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture dataset_size images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Frame capture failed.")
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
