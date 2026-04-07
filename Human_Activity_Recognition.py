# =====================================
# INSTALL (run once)
# =====================================
# pip install streamlit opencv-python tensorflow joblib numpy

# =====================================
# IMPORTS
# =====================================
import streamlit as st
import numpy as np
import cv2
import joblib
import tempfile
import tensorflow as tf

# =====================================
# CONFIG
# =====================================
IMG_SIZE = 128
MAX_FRAMES = 10

# =====================================
# LOAD MODEL + ENCODER
# =====================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("baseline_video_model.h5")
    encoder = joblib.load("video_label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# =====================================
# FRAME EXTRACTION
# =====================================
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    # Padding if fewer frames
    if len(frames) < MAX_FRAMES:
        for _ in range(MAX_FRAMES - len(frames)):
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames)

# =====================================
# UI
# =====================================
st.title("🎥 Human Activity Recognition (Video)")
st.write("Upload a video to detect the activity")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    # Save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    if st.button("Predict Activity"):

        with st.spinner("Processing video..."):

            frames = extract_frames(tfile.name)
            frames = np.expand_dims(frames, axis=0)

            preds = model.predict(frames)
            pred_class = np.argmax(preds)
            confidence = np.max(preds)

            label = encoder.inverse_transform([pred_class])[0]

        st.success(f"Predicted Activity: **{label}**")
        st.info(f"Confidence: {confidence:.2f}")

        # Show probabilities
        st.subheader("Class Probabilities")
        probs = preds[0]

        for i, p in enumerate(probs):
            st.write(f"{encoder.classes_[i]}: {p:.2f}")
