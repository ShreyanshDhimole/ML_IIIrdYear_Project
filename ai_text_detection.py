import json
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dotenv import load_dotenv
import os
load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "model_metadata.json"
HF_MODEL_ID = "roberta-base-openai-detector"


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


@st.cache_resource
def load_artifacts():
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID,ignore_mismatched_sizes = True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    threshold = float(metadata.get("threshold", 0.5))
    max_length = int(metadata.get("max_length", 512))
    id2label = getattr(model.config, "id2label", {}) or {}

    ai_label_index = 1
    for index, label in id2label.items():
        normalized_label = str(label).strip().lower()
        if any(token in normalized_label for token in ["ai", "generated", "fake"]):
            ai_label_index = int(index)
            break

    return {
        "mode": "huggingface",
        "tokenizer": tokenizer,
        "model": model,
        "metadata": metadata,
        "threshold": threshold,
        "max_length": max_length,
        "device": device,
        "model_source": HF_MODEL_ID,
        "id2label": id2label,
        "ai_label_index": ai_label_index,
    }


def analyze_text(text, artifacts):
    cleaned_text = normalize_text(text)
    encoded = artifacts["tokenizer"](
        cleaned_text,
        truncation=True,
        max_length=artifacts["max_length"],
        padding=False,
        return_tensors="pt",
    )
    encoded = {key: value.to(artifacts["device"]) for key, value in encoded.items()}

    with torch.no_grad():
        logits = artifacts["model"](**encoded).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    ai_probability = float(probabilities[artifacts["ai_label_index"]].item())

    threshold = artifacts["threshold"]
    prediction = 1 if ai_probability >= threshold else 0
    confidence = ai_probability if prediction == 1 else 1 - ai_probability
    return prediction, ai_probability, confidence, cleaned_text


st.set_page_config(
    page_title="AI Text Detector",
    page_icon="AI",
    layout="centered",
)

st.title("AI-Generated Text Detector")
st.caption("Transformer-based classifier trained on your essay dataset.")

try:
    artifacts = load_artifacts()
except Exception as error:
    st.error(f"Failed to load model artifacts: {error}")
    st.stop()


placeholder = "Paste a reasonably long paragraph or essay excerpt for analysis."
user_input = st.text_area(
    "Enter text to analyze",
    height=280,
    placeholder=placeholder,
)

col1, col2 = st.columns([1, 1])
analyze_clicked = col1.button("Analyze Text", use_container_width=True, type="primary")
clear_clicked = col2.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if analyze_clicked:
    if not user_input.strip():
        st.warning("Enter some text before running the detector.")
    else:
        prediction, ai_probability, confidence, cleaned_text = analyze_text(
            user_input, artifacts
        )

        st.subheader("Result")
        if prediction == 1:
            st.error("Likely AI-generated")
        else:
            st.success("Likely human-written")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("AI probability", f"{ai_probability:.2%}")
        metric_col2.metric("Model confidence", f"{confidence:.2%}")
        metric_col3.metric("Words", f"{len(cleaned_text.split())}")

        if len(cleaned_text.split()) < 80:
            st.warning("Short samples are harder to classify reliably.")

