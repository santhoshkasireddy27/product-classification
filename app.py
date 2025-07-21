import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("model.keras")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# UI
st.title("Text Classification App")

user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    if user_input:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=100)  # adjust maxlen if needed
        prediction = model.predict(padded)
        label = label_encoder.inverse_transform([prediction.argmax()])[0]
        st.success(f"Predicted Label: {label}")
    else:
        st.warning("Please enter some text to classify.")
