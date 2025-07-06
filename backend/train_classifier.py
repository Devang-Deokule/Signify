import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import compress

# Load the data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check class distribution
label_counts = Counter(labels)
print("Class distribution before filtering:", label_counts)

# Keep only classes with >= 2 samples
valid_classes = {label for label, count in label_counts.items() if count >= 2}
if len(valid_classes) < len(label_counts):
    print("[WARNING] Some classes have less than 2 samples and will be removed.")

# Filter data & labels
mask = [lbl in valid_classes for lbl in labels]
filtered_data = np.asarray(list(compress(data, mask)))
filtered_labels = np.asarray(list(compress(labels, mask)))

if len(filtered_labels) == 0:
    print("[FATAL] No class has enough samples (min 2). Please collect more data.")
    exit()

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    filtered_data, filtered_labels, test_size=0.2, shuffle=True, stratify=filtered_labels
)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print(f"{accuracy * 100:.2f}% of samples were classified correctly!")

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("[INFO] Model saved as 'model.p'")
