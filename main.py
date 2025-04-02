import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("cnn_model.h5")

classes = ["airplane", "automobile", "bird", "cat", "deer", 
           "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(image):
    image = image.resize((32, 32)) 
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0) 
    return image

st.markdown("""
    <style>
        body {
            background-color: #3E2723;
            text-align: center;
            font-family: 'Arial', sans-serif;
            color: white;
        }
        .container {
            max-width: 500px;
            margin: auto;
            background: #5D4037;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.2);
        }
        .button {
            background-color: #D84315;
            color: white;
            padding: 12px;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            cursor: pointer;
            margin-top: 15px;
            width: 100%;
        }
        .button:hover {
            background-color: #BF360C;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="container">', unsafe_allow_html=True)

st.title("🖼️ Object Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        predicted_class = classes[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.subheader("🔍 Prediction Result:")
        st.write(f"**Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)
