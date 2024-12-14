import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('trained_model.h5')

def process_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convert the image to grayscale
    img = cv2.resize(img, (150, 150)) #150x150 is the size of the input image
    img = img / 255.0
    img = img.reshape(1, 150, 150, -1)
    return img

st.title("Pneumonia Detection App")
st.write('This is a simple web app to predict Pneumonia from X-ray images')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    #Upload image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8) #bytearray is used to convert the image to bytes
    image = cv2.imdecode(file_bytes, 1) # 1 means load color image

    #Display image
    st.image(image, cpation = 'Uploaded image', use_column_width = True)

    #Process image and predict
    processed_image = process_image(image)
    prediction = model.predict(processed_image)
    class_names = ['Pneumonia', 'Normal']

    #Show prediction
    st.write(f"Prediction: **{class_names[int(prediction[0][0] > 0.5)]}**") #If the prediction is greater than 0.5, it is Pneumonia
    st.write(f"Probability: {prediction[0][0]:.2f}")