import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import gdown
import os

# Load model from saved file
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

# Function to make predictions
def predict_disease(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# Streamlit UI
st.title("Plant Disease Detection")

st.write("Upload an image of a plant leaf to check for disease:")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    model = load_trained_model()
    st.write("Classifying... Please wait.")
    prediction = predict_disease(uploaded_file, model)
    class_names = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Prediction: {predicted_class}")
