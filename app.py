import gdown
from keras.models import load_model
import streamlit as st
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import os

# Function to download the model from Google Drive
def download_model():
    url = 'https://drive.google.com/uc?export=download&id=1uvAp_I30bpXBt3nHQfQiow7MoGZXsJYx'  # Your Google Drive File ID
    output = 'plant_disease_model.h5'  # Save as plant_disease_model.h5
    gdown.download(url, output, quiet=False)

# Function to load the trained model
def load_trained_model():
    try:
        download_model()  # Download model if not already present
        model = load_model("plant_disease_model.h5")  # Load the model
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

# Function to preprocess the input image (resize and convert to array)
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (256, 256))  # Resize image to match model input
        img = img_to_array(img)  # Convert image to array
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize pixel values
        return img
    except Exception as e:
        print("Error preprocessing image:", e)
        return None

# Streamlit interface setup
st.title("Plant Disease Detection")
st.write("Upload a plant leaf image and get predictions for common diseases.")

# Upload image using Streamlit's file uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the trained model
    model = load_trained_model()
    
    if model:
        # Preprocess the uploaded image
        processed_img = preprocess_image(image_path)

        # Make a prediction using the trained model
        if processed_img is not None:
            prediction = model.predict(processed_img)

            # Decode the prediction (for example, print predicted class)
            classes = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
            predicted_class = classes[np.argmax(prediction)]

            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            st.write(f"Prediction: {predicted_class}")
        else:
            st.write("Error processing the image for prediction.")
    else:
        st.write("Error loading the model.")
else:
    st.write("Please upload an image to get started.")
