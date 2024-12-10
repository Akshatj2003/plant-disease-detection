import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import requests
from io import BytesIO
from PIL import Image
import gdown

# Function to download model from Google Drive (if not already present)
def download_file_from_google_drive(file_id, destination):
    URL = f'https://drive.google.com/uc?id={file_id}'
    r = requests.get(URL, stream=True)
    if r.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
    else:
        print(f"Failed to download file: {r.status_code}")

# Mount Google Drive if working locally in Colab
from google.colab import drive
drive.mount('/content/drive')

# Use Google Drive Path for loading datasets
dataset_path = '/content/drive/MyDrive/plant-disease-detection'

# Replace these with your actual file ids from Google Drive
model_file_id = 'your_model_file_id_here'

# Download model if not already present in the folder
if not os.path.exists("plant_disease_model.h5"):
    download_file_from_google_drive(model_file_id, 'plant_disease_model.h5')

# Load the trained model
def load_trained_model():
    return load_model('plant_disease_model.h5')

model = load_trained_model()

# Define your labels (mapping for the disease classes)
labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Function to prepare the image for prediction
def prepare_image(img_path):
    img = Image.open(img_path)
    img = img.resize((256, 256))  # Resize image to match model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)

# Streamlit app interface
st.title("Plant Disease Detection")

# Upload image for prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Prepare the image for prediction
    image_path = uploaded_file.name
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Prepare image
    prepared_image = prepare_image(image_path)

    # Make prediction
    prediction = model.predict(prepared_image)
    predicted_class = np.argmax(prediction)

    # Display prediction
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {labels[predicted_class]}')
