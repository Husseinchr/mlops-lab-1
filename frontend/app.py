"""Upload an image to the Food-11 inference service."""

import os

import requests
import streamlit as st

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://127.0.0.1:8000")
st.title("Food-11 classifier")
uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    st.image(uploaded, width=300)
    try:
        response = requests.post(
            f"{INFERENCE_URL}/predict",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        st.write(f"**Prediction:** {result['category']} ({result['confidence']:.1%})")
    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(f"Unable to classify the image: {error}")
