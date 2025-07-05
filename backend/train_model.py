import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set constants
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 1
DATA_DIR = r"C:\Users\Devang\OneDrive\Desktop\sign-language-web\backend\dataset\asl_alphabet_train\asl_alphabet_train"
# Create data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# Print class names
print("Classes found:", train_data.class_indices)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save the model
model.save("asl_model.h5")
print("Model trained and saved as asl_model.h5")