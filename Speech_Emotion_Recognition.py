import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tempfile

# ==============================
# LOAD MODEL
# ==============================
import tensorflow as tf

import joblib
encoder = joblib.load("encoder.pkl")
labels = encoder.classes_
model = tf.keras.models.load_model("final_model.h5")

# Replace with your actual labels (IMPORTANT)

# ==============================
# FEATURE EXTRACTION (SAME AS TRAINING)
# ==============================
def extract_from_audio(audio, sr):

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    #mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    #mel = librosa.power_to_db(mel)
    delta = librosa.feature.delta(mfcc)

    combined = np.vstack((mfcc, delta)).T

    if combined.shape[0] < 130:
        pad = np.zeros((130 - combined.shape[0], combined.shape[1]))
        combined = np.vstack((combined, pad))
    else:
        combined = combined[:130]

    return combined


# ==============================
# STREAMLIT UI
# ==============================
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a WAV file to detect emotion")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load audio
    audio, sr = librosa.load(tmp_path, duration=3)

    # Extract features
    features = extract_from_audio(audio, sr)
    features = np.expand_dims(features, axis=0)

    # Predict
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.audio(uploaded_file, format='audio/wav')

    st.subheader("Prediction:")
    st.write(f"Emotion: **{labels[predicted_class]}**")
    st.write(f"Confidence: **{confidence:.2f}**")
