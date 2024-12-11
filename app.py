import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import requests
from io import BytesIO

# Load the trained model and labels
model = load_model("model/trained_model.h5")
with open("model/labels.txt", "r") as f:
    labels = f.read().splitlines()

# Helper function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Streamlit app layout
st.title("Plant Disease Classification")
st.write("Upload an image to predict the plant disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_label = labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_label}")
    st.info(f"Confidence: {confidence:.2f}%")
