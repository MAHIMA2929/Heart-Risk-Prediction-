import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import base64
import pickle

# ---------------- Page Config ----------------
st.set_page_config(page_title="Heart Risk Prediction", layout="wide")

# ---------------- Background Image ----------------
def get_base64(file):
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

img = get_base64("background.jpg")

if img:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("<h1 style='text-align:center;'>❤️ Heart Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:lightgray;'>Enter your health details</h4>", unsafe_allow_html=True)

st.write("")

# ---------------- Load Model ----------------
use_model = False
try:
    model = pickle.load(open("model.pkl", "rb"))
    use_model = True
except:
    st.info("Using basic prediction logic")

# ---------------- Input Section ----------------
st.subheader("Patient Health Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, value=40)
    bmi = st.number_input("BMI", 10.0, 50.0, value=25.0)
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    glucose = st.number_input("Glucose Level", 50, 300, value=120)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])

heart_val = 1 if heart_disease == "Yes" else 0
hypertension_val = 1 if hypertension == "Yes" else 0

st.write("")

# ---------------- Prediction Button ----------------
if st.button("🚀 Predict Heart Risk"):

    # ---------------- Risk Logic ----------------
    if use_model:
        try:
            data = np.array([[age, hypertension_val, heart_val, glucose, bmi]])
            risk = model.predict_proba(data)[0][1] * 100
        except:
            risk = 50
    else:
        risk = (age * 0.3) + (bmi * 0.4) + (glucose * 0.2)

        if hypertension_val == 1:
            risk += 20

        if heart_val == 1:
            risk += 20

        risk = min(max(risk / 2, 0), 100)

    # ---------------- Result ----------------
    st.write("")

    if risk >= 70:
        st.error(f"⚠️ High Risk: {risk:.1f}%")
    elif risk >= 40:
        st.warning(f"⚠️ Moderate Risk: {risk:.1f}%")
    else:
        st.success(f"✅ Low Risk: {risk:.1f}%")

    st.write("")

    # ---------------- Graph Row 1 ----------------
    colA, colB = st.columns(2)

    # Gauge Chart
    with colA:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={'text': "Heart Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))

        st.plotly_chart(fig_gauge, use_container_width=True)

    # Bar Chart
    with colB:
        fig_bar = px.bar(
            x=["Low Risk", "High Risk"],
            y=[100-risk, risk],
            labels={'x': 'Category', 'y': 'Percentage'},
            title="Risk Distribution"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------- Graph Row 2 ----------------
    colC, colD = st.columns(2)

    # Pie Chart
    with colC:
        fig_pie = px.pie(
            names=["Low Risk", "High Risk"],
            values=[100-risk, risk],
            title="Risk Share"
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    # Health Data Chart
    with colD:
        fig_health = px.bar(
            x=["Age", "BMI", "Glucose"],
            y=[age, bmi, glucose],
            title="User Health Data"
        )

        st.plotly_chart(fig_health, use_container_width=True)