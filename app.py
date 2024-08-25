import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import keras
st.markdown(
    """
    <style>
    /* Background color for main content */
    .stApp {
        background-color: #e6f7ff;
        color:black;
    }

    /* Custom CSS for the sidebar */
    [data-testid="stSidebar"] {
        background-color: lightcoral;
        padding: 20px;
        border-top-right-radius:10px;
        border-bottom-right-radius:10px;
    }

    /* Header style in sidebar */
    [data-testid="stSidebar"] h2 {
        color: navy;
    }

    /* Text style in sidebar */
    [data-testid="stSidebar"] p {
        color: #333;
        font-size: 16px;
    }

    /* Divider line in sidebar */
    [data-testid="stSidebar"] hr {
        margin: 20px 0;
        border: none;
        border-top: 2px solid #007acc;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load the custom .keras model
model = keras.models.load_model("model_saved.keras")

# Streamlit app title
st.title("🐱 Cat vs. 🐶 Dog Classifier")

# Sidebar for input
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Add some instructions
st.markdown("""
This app uses a Convolutional Neural Network (CNN) model to classify images as either a cat or a dog. 
Simply upload an image, and the model will predict the class.
""")

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image to match the model's input shape
    img = img.resize((224, 224))  # Adjust size based on your model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Check the model's expected input shape
    input_shape = model.input_shape
    if len(input_shape) == 2:
        # Flatten the image array if the model expects a flat input
        img_array = img_array.reshape(1, -1)
    elif len(input_shape) == 4 and input_shape[1:] != (224, 224, 3):
        # If the model expects a different input size, adjust the image size accordingly
        img = img.resize((input_shape[1], input_shape[2]))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

    # Add a spinner while processing
    with st.spinner('Classifying...'):
        preds = model.predict(img_array)

    # Assuming binary classification (cat vs. dog)
    confidence = preds[0][0]  # Extract the scalar value
    if preds[0][0]==1:
        label = "Dog 🐶"
    else:
        label = "Cat 🐱"
        confidence = 1 - confidence

    # Display the result with styling
    st.markdown(f"## Prediction: **{label}**")
    # Option to upload another image
    st.sidebar.write("Upload another image to classify again.")

else:
    st.info("Please upload an image using the sidebar.")

