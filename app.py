import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained model and label encoder
with open("hand_sign_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
label_encoder = data["label_encoder"]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandSignPredictor(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False,
                                     max_num_hands=1,
                                     min_detection_confidence=0.7,
                                     min_tracking_confidence=0.7)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                # Prediction
                prediction = model.predict([landmarks])[0]
                sign_name = label_encoder.inverse_transform([prediction])[0]

                # Display prediction on frame
                cv2.putText(image, f"Sign: {sign_name}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image

st.set_page_config(page_title="Sign Language Detector", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.3em;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

# Main title and stream
st.markdown('<div class="title">🖐️ Real-Time Sign Language Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Using your webcam to recognize ASL alphabet signs</div>', unsafe_allow_html=True)
st.markdown("---")

webrtc_streamer(key="hand-sign", video_transformer_factory=HandSignPredictor)