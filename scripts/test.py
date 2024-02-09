import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def preprocess_image(image_path, target_size=(224, 224)):
    if os.path.exists(image_path):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        return img_array
    else:
        print(f"Image not found: {image_path}")
        return None

# Load CSV file
data = pd.read_csv("PlantCLEF2022_web_training_metadata.csv", delimiter=";")



# List IDs in the directory
directory_ids = os.listdir("/Users/istalter/Desktop/images")
directory_ids = [int(directory_id) for directory_id in directory_ids if directory_id.isdigit()]

num_classes_csv = data['species'].nunique()

# Get number of unique directory IDs
num_classes_dirs = len(directory_ids)

# Assign the maximum of the two values to NUM_CLASSES
NUM_CLASSES = max(num_classes_csv, num_classes_dirs)

# Filter data based on directory IDs
filtered_data = data[data['classid'].isin(directory_ids)]

# Filter out None values and preprocess images
X = []
for img_path in filtered_data['image_path']:
    img_array = preprocess_image(img_path)
    if img_array is not None:
        X.append(img_array)
    else:
        print(f"Failed to preprocess image: {img_path}")

# Convert list to numpy array
X = np.array(X)

# Print the number of images processed
print("Number of images processed:", len(X))

# Assuming you have labels and want to train a model

# Load labels
y = filtered_data['species']

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = models.Sequential([
    layers.Flatten(input_shape=(224, 224, 3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
