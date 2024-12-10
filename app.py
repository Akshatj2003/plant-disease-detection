import streamlit as st
import os
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import requests
from io import BytesIO
from PIL import Image

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(URL)
    with open(destination, 'wb') as f:
        f.write(response.content)

# Function to download dataset (using Google Drive links)
def download_datasets():
    dataset_links = [
        "1x_6Ksz-FdFcsmnRKnrUEQEv6hEFMiWrb",  # Dataset 1
        "1CSaoJ-iONMXEBPS5J8DBLQUAHa0Ud9xM",  # Dataset 2
        "1LIo8adY1KimV0BNCwVZlqOS-Xz1frVV5"   # Dataset 3
    ]
    # For each link, download the dataset (assuming it's a folder, download files manually or use API)
    # You might need to manually download and place them in the app directory or process them via Google API.

# Load the model
def load_trained_model():
    model_file = "plant_disease_model.h5"
    if not os.path.exists(model_file):
        model_file_id = "YOUR_MODEL_FILE_ID"  # Replace with your actual model file ID
        download_file_from_google_drive(model_file_id, model_file)
    
    model = load_model(model_file)
    return model

# Load the model
model = load_trained_model()

# Function to preprocess image for prediction
def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Resize to the input size expected by the model
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image
    return img

# Function to predict the disease of the plant
def predict_disease(img_path):
    prepared_img = prepare_image(img_path)
    prediction = model.predict(prepared_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
    return class_names[predicted_class]

# Set up Streamlit interface
st.title("Plant Disease Detection")
st.write("Upload an image of a plant leaf to predict its disease")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image("uploaded_image.jpg", caption="Uploaded Image", use_column_width=True)
    
    # Predict the disease
    prediction = predict_disease("uploaded_image.jpg")
    st.write(f"Predicted Disease: {prediction}")
