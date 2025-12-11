import streamlit as st
from tensorflow.keras.models import load_model 
from PIL import Image
import numpy as np
import os

@st.cache_resource
def load_model_cached():
    model_path = '/home/saurav-neupane/Downloads/archive (1)/Dataset - train+val+test/Trained_Eye_disease_model.keras'
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'.")
        return None
    return load_model(model_path)

model = load_model_cached()

class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

st.title("Eye Disease Prediction Web App 👁️")
st.write("Upload an eye image to predict the disease.")

if model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Eye Image', use_container_width=True)

    if st.button("Predict"):
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized)
        image_array = image_array / 255.0 
        image_array = np.expand_dims(image_array, axis=0)

        prediction_probs = model.predict(image_array)[0]
        pred_class = np.argmax(prediction_probs)
        confidence = prediction_probs[pred_class] * 100

        st.subheader("Prediction Results")
        st.success(f"Predicted Eye Disease: **{class_names[pred_class]}**")
        st.info(f"Confidence: {confidence:.2f}%")

        st.write("Raw Probabilities:")
        for i, name in enumerate(class_names):
            st.write(f"- {name}: {prediction_probs[i]*100:.2f}%")
