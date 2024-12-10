import streamlit as st
import numpy as np
import os
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import gdown

# Function to download model from Google Drive
def download_model():
    url = 'https://drive.google.com/uc?id=1SKkEKZVZtXGsnGc_-Gk2vpxRDBt2QwHq'  # Replace with the actual model file link
    output = 'plant_disease_model.h5'
    gdown.download(url, output, quiet=False)
    st.write(f"Downloaded model {output}")

# Function to preprocess and predict image
def convert_image_to_array(image_file):
    try:
        image = Image.open(image_file)
        image = image.resize((256, 256))
        image = np.array(image)
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        st.error(f"Error in image processing: {e}")
        return None

# Load model
@st.cache_resource
def load_trained_model():
    download_model()  # This will download the model before loading
    model = load_model("plant_disease_model.h5")
    return model

# Display title and description
st.title("Plant Disease Detection")
st.write("This application detects plant diseases using a trained CNN model.")

# Load the trained model
model = load_trained_model()

# Upload image through Streamlit's file uploader
uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    
    # Preprocess image and make prediction
    image_array = convert_image_to_array(uploaded_file)
    
    if image_array is not None:
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)
        
        classes = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
        result = classes[predicted_class[0]]
        
        st.success(f"Prediction: {result}")
