import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown
import os
import cv2

# Function to download the model from Google Drive
def download_model():
    model_file = "trained_model.h5"
    if not os.path.exists(model_file):
        url = 'https://drive.google.com/uc?export=download&id=your_model_file_id'
        gdown.download(url, model_file, quiet=False)
    return model_file

# Load the trained model
model_path = download_model()
model = load_model(model_path)

# Class labels
class_names = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Function to preprocess and predict the image
def preprocess_image(image):
    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("Plant Disease Detection")
st.write("Upload a plant leaf image to detect the disease.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Prediction: {predicted_class}")
