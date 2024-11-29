import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_model():
    model_path = 'malaria_detection_model.keras'
    model = tf.keras.models.load_model(model_path)
    return model

# Preprocess the image
def preprocess_image(image):
    # Resize image to match training input size
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction function
def predict_malaria(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_names = ['Parasitized', 'Uninfected']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

def main():
    st.title('Malaria Cell Image Detection')
    
    # Sidebar for model information
    st.sidebar.header('Model Information')
    st.sidebar.write('ResNet-based Malaria Detection Model')
    st.sidebar.write('Input: Cell Microscopic Images')
    st.sidebar.write('Output: Malaria Infection Status')

    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a cell microscopic image", 
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)

        # Make prediction
        if st.button('Detect Malaria'):
            try:
                # Predict
                predicted_class, confidence = predict_malaria(image, model)
                
                # Display results
                st.subheader('Prediction Results')
                st.write(f'Predicted Class: {predicted_class}')
                st.write(f'Confidence: {confidence:.2f}%')

                # Visualization of prediction
                if predicted_class == 'Parasitized':
                    st.warning('⚠️ Malaria Parasite Detected')
                else:
                    st.success('✅ No Malaria Parasite Detected')

            except Exception as e:
                st.error(f"Error making prediction: {e}")

    # Additional information
    st.markdown("""
    ### How to Use
    1. Upload a microscopic cell image
    2. Click 'Detect Malaria'
    3. View the prediction result
    
    ### About the Model
    - Trained on cell microscopic images
    - Uses ResNet architecture
    - Classifies images as Parasitized or Uninfected
    """)

if __name__ == '__main__':
    main()
