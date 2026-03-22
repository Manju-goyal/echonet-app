import streamlit as st
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model

st.title("💓 Heart EF Prediction App")

# 🔥 Load model safely
@st.cache_resource
def load_my_model():
    if not os.path.exists("model.keras"):
        st.write("Downloading model... ⏳")
        url = "https://drive.google.com/uc?id=1wyUhTvWsos6YQJ69bv2kQNlS3nueACvN"
        gdown.download(url, "model.keras", quiet=False)

    st.write("Loading real model... 🤖")
    model = load_model("model.keras",compile=False)
    return model

model = load_my_model()

# 🎥 Video processing
def load_video(path, max_frames=16):
    cap = cv2.VideoCapture(path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    while len(frames) < max_frames:
        frames.append(frames[-1])

    return np.array(frames)

# 📂 Upload UI
uploaded_file = st.file_uploader("Upload Echo Video", type=["mp4", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing video... 🎥")

    frames = load_video("temp_video.mp4")

    if frames is None:
        st.error("❌ Video read nahi hua")
    else:
        frames = np.expand_dims(frames, axis=0)

        st.write("Predicting... 🤖")

        prediction = model.predict(frames)[0][0]

        st.success(f"💓 Predicted EF: {prediction:.2f}")
