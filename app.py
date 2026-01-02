import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Silent Dropout Risk Prediction",
    page_icon="⚠️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main-header {font-size: 3rem; color: #1e3a8a; text-align: center;}
.sub-header {text-align: center; color: #6b7280; margin-bottom: 2rem;}
.risk-card {padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;}
.high-risk {background:#fee2e2; border-left:4px solid #dc2626;}
.medium-risk {background:#fef3c7; border-left:4px solid #d97706;}
.low-risk {background:#dcfce7; border-left:4px solid #16a34a;}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<h1 class="main-header">⚠️ Silent Dropout Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered patient retention risk assessment</p>', unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
def load_models():
    files = ["svm_model.pkl", "scaler.pkl", "label_encoder.pkl", "feature_columns.pkl"]
    for f in files:
        if not os.path.exists(f):
            st.error(f"Missing file: {f}")
            st.stop()

    model = pickle.load(open("svm_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
    return model, scaler, le, feature_columns

model, scaler, le, feature_columns = load_models()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write("**Model:** SVM")
    st.write("**Output:** Low / Medium / High Risk")
    st.write("**Purpose:** Silent patient dropout detection")

# ---------------- INPUT FORM ----------------
st.header("🧾 Patient Assessment")

col1, col2 = st.columns(2)

with col1:
    Days_Since_Last_Contact = st.number_input("Days Since Last Contact", 0, 365, 10)
    Expected_Gap_Between_Visits_days = st.number_input("Expected Gap Between Visits (days)", 0, 365, 7)
    Days_Late_Follow_Up = st.number_input("Days Late for Follow-Up", 0, 365, 5)

with col2:
    Medicine_Refill_Delay_days = st.number_input("Medicine Refill Delay (days)", 0, 180, 2)
    Missed_Lab_Tests = st.number_input("Missed Lab Tests", 0, 10, 0)
    Silent_Dropout_Score = st.slider("Silent Dropout Score", 0.0, 10.0, 3.0, 0.1)

input_values = {
    "Days_Since_Last_Contact": Days_Since_Last_Contact,
    "Expected_Gap_Between_Visits_days": Expected_Gap_Between_Visits_days,
    "Medicine_Refill_Delay_days": Medicine_Refill_Delay_days,
    "Missed_Lab_Tests": Missed_Lab_Tests,
    "Days_Late_Follow_Up": Days_Late_Follow_Up,
    "Silent_Dropout_Score": Silent_Dropout_Score
}

# ---------------- PREDICT ----------------
st.markdown("---")
predict_button = st.button("🔮 Predict Risk Level", use_container_width=True)

if predict_button:
    try:
        with st.spinner("🤖 Running model..."):

            # Build DataFrame
            input_df = pd.DataFrame([input_values])

            # Ensure correct feature order
            input_df = input_df.reindex(columns=feature_columns).fillna(0).astype(float)

            # Scale correctly
            if hasattr(scaler, "feature_names_in_"):
                scaler_cols = list(scaler.feature_names_in_)
                scaled = scaler.transform(input_df[scaler_cols])
                scaled_df = pd.DataFrame(scaled, columns=scaler_cols, index=input_df.index)
                input_df = scaled_df.reindex(columns=feature_columns).fillna(0)
            else:
                input_df.loc[:, feature_columns] = scaler.transform(input_df[feature_columns])

            X = input_df.values

            # Predict
            raw_pred = model.predict(X)

            try:
                risk_level = le.inverse_transform(raw_pred)[0]
            except:
                risk_level = str(raw_pred[0])

            # Confidence
            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(X))
            elif hasattr(model, "decision_function"):
                score = model.decision_function(X)
                score = score[0] if isinstance(score, (list, np.ndarray)) else score
                confidence = float(1 / (1 + np.exp(-score)))

        # ---------------- OUTPUT ----------------
        st.header("📊 Prediction Results")

        if risk_level.lower() == "high":
            st.markdown(f"<div class='risk-card high-risk'>🔴 <b>HIGH RISK</b><br>Confidence: {confidence:.1%}</div>", unsafe_allow_html=True)
        elif risk_level.lower() == "medium":
            st.markdown(f"<div class='risk-card medium-risk'>🟡 <b>MEDIUM RISK</b><br>Confidence: {confidence:.1%}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='risk-card low-risk'>🟢 <b>LOW RISK</b><br>Confidence: {confidence:.1%}</div>", unsafe_allow_html=True)

        # Risk score
        risk_score = (
            Days_Since_Last_Contact * 0.2 +
            Days_Late_Follow_Up * 0.3 +
            Medicine_Refill_Delay_days * 0.25 +
            Missed_Lab_Tests * 5 +
            Silent_Dropout_Score * 2
        )

        st.metric("Risk Score", f"{risk_score:.1f}/100")
        st.metric("Assessment Date", datetime.now().strftime("%Y-%m-%d"))

        # Recommendations
        st.markdown("### 💡 Recommendations")
        if risk_level.lower() == "high":
            st.error("Immediate follow-up within 24 hours.")
        elif risk_level.lower() == "medium":
            st.warning("Follow-up recommended within 3 days.")
        else:
            st.success("Patient stable. Continue normal care.")

        # ---------------- BAR CHART ----------------
        st.markdown("### 🔍 Risk Factors")
        fig, ax = plt.subplots(figsize=(9, 4))
        values = list(input_values.values())
        labels = list(input_values.keys())
        ax.barh(labels, values)
        ax.set_title("Input Feature Values")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center><small>Silent Dropout Risk Prediction System | ML Powered</small></center>",
    unsafe_allow_html=True
)
