import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("plant_disease_model.h5")  # Ensure the model is in the same directory

# Preprocessing Function
def preprocess_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).resize((256, 256))  # Resize to match model input
        image = img_to_array(image) / 255.0  # Normalize pixel values
        return np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        st.error(f"Error in processing the image: {e}")
        return None

# Prediction Function
def predict_disease(image_array, model):
    classes = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
    prediction = model.predict(image_array)
    return classes[np.argmax(prediction)]

# Streamlit App
st.title("Plant Disease Identification")
st.write("Upload an image of a plant leaf, and the model will predict its disease.")

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and Predict
    processed_image = preprocess_image(uploaded_file)
    if processed_image is not None:
        model = load_trained_model()
        prediction = predict_disease(processed_image, model)
        st.success(f"**Predicted Disease:** {prediction}")
    else:
        st.error("Error in processing the image. Please try again.")
