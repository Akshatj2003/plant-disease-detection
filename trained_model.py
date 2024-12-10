import numpy as np
import os
from os import listdir
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image

# Data Loading and Preprocessing
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, (256, 256))
            return img_to_array(image)
        else:
            return np.array([])  # Empty array if image loading fails
    except Exception as e:
        print(f"Error: {e}")
        return None

# Directory and Labels
dir = r"your_dataset_path_here"  # Change to your dataset path
root_dir = listdir(dir)
all_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
binary_labels = [0, 1, 2]

# Load Data
image_list, label_list = [], []
temp = -1
for directory in root_dir:
    plant_image_list = listdir(f"{dir}/{directory}")
    temp += 1
    for files in plant_image_list:
        image_path = f"{dir}/{directory}/{files}"
        image_list.append(convert_image_to_array(image_path))
        label_list.append(binary_labels[temp])

# Convert Data to Arrays
image_list = np.array(image_list, dtype=np.float16) / 255.0  # Normalize pixel values
label_list = np.array(label_list)
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state=10)

# One-Hot Encoding for Labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the Model
model = Sequential([
    Conv2D(32, (3, 3), padding="same", input_shape=(256, 256, 3), activation="relu"),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(16, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")  # 3 classes
])

# Compile the Model
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

# Train the Model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=128)

# Save the Model
model.save('plant_disease_model.h5')  # Save for deployment
print("Model saved as plant_disease_model.h5")
