import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Path to your dataset folder with subfolders for each class label
DATASET_DIR = "dataset"  # Change this to your dataset folder path

# Image input size
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% for validation
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Save class_indices (label → index) to JSON for later use in inference
class_indices = train_generator.class_indices
print("Class Indices:", class_indices)
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

# Define CNN model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_indices), activation='softmax')  # Output layer with softmax for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Show model summary
model.summary()

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# Create directory if not exists and save the trained model
os.makedirs('models', exist_ok=True)
model.save('models/asl_model.h5')
print("Model saved as models/asl_model.h5")
