import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the model
model = load_model("trained_model.h5")
labels = ["Corn-Common_rust", "Potato-Early_blight", "Tomato-Bacterial_spot"]

st.title("Plant Disease Classification")
st.write("Upload an image of a leaf to classify the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = load_img(uploaded_file, target_size=(256, 256))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    prediction = model.predict(image)
    predicted_label = labels[np.argmax(prediction)]

    st.write(f"Prediction: **{predicted_label}**")
