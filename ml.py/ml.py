import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

st.header('Image Classification Model')

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model("D:\projml\step1success.ipynb")

model = load_model()

# Define the classes
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Image classification function
def classify_image(image_path):
    img_height = 180
    img_width = 180

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image data

    # Predict the class of the image
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction)

    return data_cat[np.argmax(score)], np.max(score) * 100

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...s", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Perform image classification when button is clicked
    if st.button('Classify'):
        class_name, confidence = classify_image(uploaded_file)
        st.write('Fruit in image is ' + class_name)
        st.write('With accuracy of ' + str(confidence))
