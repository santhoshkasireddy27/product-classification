import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model
model = tf.keras.models.load_model("model.h5")

# Load tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load label encoder
with open("label_encoder.pickle", "rb") as handle:
    label_encoder = pickle.load(handle)

# Constants
MAX_SEQUENCE_LENGTH = 100  # same as used during training

# Title
st.title("🛍️ ShopEase Product Category Classifier")

# Description
st.write("Classify products into categories (e.g., Electronics, Clothing, etc.) based on their description and attributes.")

# Input
user_input = st.text_area("Enter product description:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a product description.")
    else:
        # Preprocess
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
        
        # Predict
        prediction = model.predict(padded)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

        st.success(f"Predicted Category: **{predicted_label[0]}**")

