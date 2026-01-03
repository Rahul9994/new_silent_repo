import streamlit as st
import pandas as pd
from datetime import date

# -------------------------------
# Dataset-aligned scoring logic
# -------------------------------
def predict_risk(X):
    silent_score = (
        0.1 * X["Medicine_Refill_Delay_days"] +
        2.0 * X["Missed_Lab_Tests"] +
        0.1 * X["Days_Late_Follow_Up"]
    )

    if silent_score.iloc[0] > 9:
        risk = "HIGH"
    elif silent_score.iloc[0] >= 4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return silent_score.iloc[0], risk

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Patient Silent Dropout Risk Predictor", layout="centered")

st.title("🩺 Patient Silent Dropout Risk Predictor")
st.markdown("Predict dropout risk using **dataset-trained clinical scoring logic**.")

# Inputs
last_follow_up = st.date_input("Last Follow-Up Date", value=date.today())
expected_gap = st.number_input("Expected Gap Between Visits (days)", min_value=0)
refill_delay = st.number_input("Medicine Refill Delay (days)", min_value=0)
days_since_contact = st.number_input("Days Since Last Contact", min_value=0)
missed_labs = st.number_input("Missed Lab Tests", min_value=0)
days_late_followup = st.number_input("Days Late for Follow-Up", min_value=0)

# Prediction
if st.button("🔍 Predict Risk"):
    input_df = pd.DataFrame({
        "Expected_Gap_Between_Visits_days": [expected_gap],
        "Medicine_Refill_Delay_days": [refill_delay],
        "Days_Since_Last_Contact": [days_since_contact],
        "Missed_Lab_Tests": [missed_labs],
        "Days_Late_Follow_Up": [days_late_followup]
    })

    score, risk_level = predict_risk(input_df)

    st.success("Prediction Complete ✅")
    st.metric("Silent Dropout Score", round(score, 2))
    st.metric("Risk Level", risk_level)

    if risk_level == "HIGH":
        st.error("🚨 High silent dropout risk — immediate intervention required.")
    elif risk_level == "MEDIUM":
        st.warning("⚠️ Moderate risk — monitor closely.")
    else:
        st.success("✅ Low risk — patient engagement is healthy.")
