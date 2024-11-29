import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load the pre-trained model
@st.cache_resource
def load_model():
    model_path = 'malaria_detection_model.keras'
    model = tf.keras.models.load_model(model_path)
    return model

# Preprocess image for prediction
def preprocess_image(image):
    # Resize image to match training input size
    img_resized = cv2.resize(np.array(image), (64, 64))
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_preprocessed = np.expand_dims(img_normalized, axis=0)
    
    return img_preprocessed

# Main Streamlit app
def main():
    st.title('Malaria Cell Detection')
    st.write('Upload a cell image to detect if it is parasitized or uninfected.')

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a cell image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Cell Image', use_column_width=True)

        # Load model
        try:
            model = load_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Preprocess image
        preprocessed_image = preprocess_image(image)

        # Predict
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class] * 100

        # Map prediction to class names
        class_names = ['Parasitized', 'Uninfected']
        result = class_names[predicted_class]

        # Display results
        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Visualization of prediction probabilities
        st.bar_chart({
            'Parasitized': prediction[0][0],
            'Uninfected': prediction[0][1]
        })

if __name__ == "__main__":
    main()