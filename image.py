import streamlit as st
from transformers import pipeline
from PIL import Image

# -------------------------------
# Load Model (Auto Download + Cache)
# -------------------------------
@st.cache_resource
def load_model():
    pipe = pipeline(
        "image-classification",
        model="jazzmacedo/fruits-and-vegetables-detector-36"
    )
    return pipe

pipe = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("Fruits & Vegetables Classifier 🍎🥦")
st.write("Upload an image and the AI will classify it")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = pipe(image)

    st.subheader("Prediction")

    top = results[0]
    st.write(f"Item: **{top['label']}**")
    st.write(f"Confidence: **{top['score']*100:.2f}%**")

    st.subheader("Top Predictions")
    for res in results[:5]:
        st.write(f"{res['label']} → {res['score']*100:.2f}%")