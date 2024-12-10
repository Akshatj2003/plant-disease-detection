import os
import cv2
import numpy as np
import gdown
import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# Helper function to download files from Google Drive
def download_file_from_drive(url, output_path):
    gdown.download(url, output_path, quiet=False)

# Paths for your dataset and model on Google Drive
model_url = 'YOUR_GOOGLE_DRIVE_MODEL_URL'  # Replace with your actual Google Drive URL for the model
dataset_url = 'YOUR_GOOGLE_DRIVE_DATASET_URL'  # Replace with your actual Google Drive URL for the dataset

# Download the model and dataset if not already available
if not os.path.exists("plant_disease_model.h5"):
    download_file_from_drive(model_url, "plant_disease_model.h5")

# Load the model
model = load_model("plant_disease_model.h5")

# Define the function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))  # Resize image to 256x256
    image = img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    return image

# Define the class names based on your dataset
class_names = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Streamlit UI
st.title("Plant Disease Detection")
st.write("Upload an image of a plant to classify its disease.")

# Upload image from user
uploaded_image = st.file_uploader("Choose a plant image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Save the uploaded image
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_image.getbuffer())
    
    # Preprocess the image
    image = preprocess_image("uploaded_image.jpg")

    # Predict the class
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Display the result
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
    st.write(f"Predicted Class: {class_names[predicted_class[0]]}")
