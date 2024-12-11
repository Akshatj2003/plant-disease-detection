import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import gdown
from PIL import Image

# Google Drive file ID for trained model
MODEL_DRIVE_ID = "YOUR_GOOGLE_DRIVE_MODEL_FILE_ID"

# Function to download the model dynamically
def download_model():
    model_file = "trained_model.h5"
    if not os.path.exists(model_file):
        st.info("Downloading model file...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}", model_file, quiet=False)
    return model_file

# Load the model
model_path = download_model()
model = load_model(model_path)

# Define class labels
CLASSES = ['Corn - Common Rust', 'Potato - Early Blight', 'Tomato - Bacterial Spot']

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0
    return image_array

# Streamlit app layout
st.title("Plant Disease Detection")
st.write("Upload an image of a leaf to detect its disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Processing...")

    # Open the uploaded file as a PIL image
    image = Image.open(uploaded_file)

    # Preprocess the image and make predictions
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = CLASSES[np.argmax(prediction)]

    # Display prediction results
    st.success(f"Prediction: {predicted_class}")
