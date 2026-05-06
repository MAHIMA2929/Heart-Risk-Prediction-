import streamlit as st
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import time
import plotly.express as px

# =========================
# 🎨 BACKGROUND
# =========================
def set_bg(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1, h2, h3, p {{
            color: white !important;
            text-shadow: 2px 2px 5px black;
        }}
        </style>
        """, unsafe_allow_html=True)
    except:
        st.warning("⚠️ bg.jpeg not found")

set_bg("bg.jpeg")

# =========================
# 🔊 SOUND FUNCTION
# =========================
def play_alert():
    try:
        with open("alert.mp3", "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()

        st.markdown(f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)
    except:
        st.warning("⚠️ alert.mp3 not found")

# ======================
# 🧠 LOAD MODEL
# ======================
@st.cache_resource
def load_my_model():
    return load_model("model/mask_model.keras")

model = load_my_model()

# ======================
# 👤 FACE DETECTOR
# ======================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ======================
# 🏷️ TITLE
# ======================
st.title("😷 Face Mask Detection System")

# ======================
# 📊 SESSION STATE
# ======================
if "mask" not in st.session_state:
    st.session_state.mask = 0
if "no_mask" not in st.session_state:
    st.session_state.no_mask = 0
if "alert_flag" not in st.session_state:
    st.session_state.alert_flag = False

# ======================
# 🔍 DETECTION FUNCTION
# ======================
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_count = len(faces)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)[0][0]

        if pred < 0.5:
            label = "Mask 😷"
            color = (0, 255, 0)
            st.session_state.mask += 1
            st.session_state.alert_flag = False
        else:
            label = "No Mask ❌"
            color = (0, 0, 255)
            st.session_state.no_mask += 1

            if not st.session_state.alert_flag:
                play_alert()
                st.session_state.alert_flag = True

        # box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        # label bg
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y-30), (x+tw, y), color, -1)

        cv2.putText(frame, label, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return frame, face_count

# ======================
# 🖼️ IMAGE DETECTION
# ======================
st.subheader("🖼️ Image Detection")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file is not None:
    img = Image.open(file)
    frame = np.array(img)

    if frame.shape[-1] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    result, count = detect(frame)

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    st.success(f"👤 Faces Detected: {count}")

# ======================
# 📷 CAMERA DETECTION
# ======================
st.subheader("📷 Camera Detection")

run = st.checkbox("Start Camera")
frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Camera not available")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                break

            frame, count = detect(frame)

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            st.write(f"👤 Live Faces: {count}")

            time.sleep(0.03)

        cap.release()

# ======================
# 📊 GRAPH (FIXED)
# ======================
st.subheader("📊 Detection Analytics")


mask_count = st.session_state.mask
no_mask_count = st.session_state.no_mask

total = mask_count + no_mask_count

if total == 0:
    st.warning("⚠️ No data available yet. Run detection first.")
else:
    df = pd.DataFrame({
        "Category": ["Mask 😷", "No Mask ❌"],
        "Count": [mask_count, no_mask_count]
    })

    st.write("### 📈 Data Table")
    st.write(df)

    fig = px.pie(
        df,
        names="Category",
        values="Count",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)