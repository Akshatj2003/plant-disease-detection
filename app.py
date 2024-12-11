import os
import gdown
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Function to download the model from Google Drive
def download_model():
    model_url = "https://drive.google.com/uc?id=MODEL_DRIVE_ID"  # Replace MODEL_DRIVE_ID with your model file ID
    model_file = "trained_model.h5"  # Local file path where the model will be saved
    
    if not os.path.exists(model_file):
        print("[INFO] Downloading trained model...")
        gdown.download(model_url, model_file, quiet=False)
    else:
        print("[INFO] Model file already exists.")
    
    return model_file

# Load the trained model
model_path = download_model()
model = load_model(model_path)

# Define the Streamlit app
st.title("Plant Disease Detection")
st.write("Upload an image to identify the plant disease")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    # Preprocess the image
    image = np.array(image.resize((256, 256))) / 255.0
    image = image.reshape(1, 256, 256, 3)
    
    # Make a prediction
    prediction = model.predict(image)
    classes = ["Corn - Common Rust", "Potato - Early Blight", "Tomato - Bacterial Spot"]
    st.write(f"Prediction: {classes[np.argmax(prediction)]}")
