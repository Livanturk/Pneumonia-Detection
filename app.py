import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras

# Modeli yükle
model = keras.models.load_model('trained_model.h5')

# Farklı formatlardaki görüntüleri işleyecek fonksiyon
def process_image(image, target_size=(150, 150), expected_channels=3):
    if len(image.shape) == 2:  # Grayscale
        if expected_channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] != expected_channels:
            if expected_channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, target_size)
    image = image / 255.0
    if expected_channels == 1:
        image = image.reshape(1, target_size[0], target_size[1], 1)
    else:
        image = image.reshape(1, target_size[0], target_size[1], expected_channels)
    return image

st.title("Pneumonia Detection App")
st.write("This is a simple web app to predict Pneumonia from X-ray images.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # Görüntüyü göster
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Görüntüyü işle ve tahmin yap
    processed_image = process_image(image)
    prediction = model.predict(processed_image)
    class_names = ["Pneumonia", "Normal"]

    st.write(f"Prediction: **{class_names[int(prediction[0][0] > 0.5)]}**")
    st.write(f"Probability: {prediction[0][0]:.2f}")
