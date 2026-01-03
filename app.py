import streamlit as st
import pandas as pd
from datetime import date

# -------------------------------
# Streamlit UI Configuration
# -------------------------------
st.set_page_config(page_title="Patient Silent Dropout Risk Predictor", layout="centered")

# -------------------------------
# INJECTING THE MOVING BACKGROUND (HTML/CSS)
# -------------------------------
st.markdown("""
<style>
    /* 1. Hide the default Streamlit white background */
    .stApp {
        background: transparent;
    }

    /* 2. The Dynamic Background Container */
    .dynamic-bg {
        position: fixed; /* Fixed ensures it stays put when scrolling */
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        
        /* Medical Image */
        background-image: url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&q=80&w=2000');
        background-size: cover;
        background-position: center;
        
        /* Animation: 8s speed as requested */
        animation: breatheAndMove 8s ease-in-out infinite alternate;
        
        z-index: -1; /* Places it BEHIND the python inputs */
    }

    /* 3. The Animation Keyframes */
    @keyframes breatheAndMove {
        0% { transform: scale(1) translate(0, 0); }
        50% { transform: scale(1.15) translate(-20px, 10px); }
        100% { transform: scale(1) translate(0, 0); }
    }

    /* 4. OPTIONAL: Make text readable against the dark background */
    h1, h2, h3, p, label, .stMarkdown {
        color: white !important;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.8); /* Adds shadow for readability */
    }
    
    /* Make input boxes slightly transparent white so labels are clear */
    .stNumberInput, .stDateInput {
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 5px;
    }
</style>

<div class="dynamic-bg"></div>
""", unsafe_allow_html=True)

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
# Application Content
# -------------------------------

st.title("🩺 Patient Silent Dropout Risk Predictor")
st.markdown("Predict dropout risk using **dataset-trained clinical scoring logic**.")

# Inputs Container
with st.container():
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

    # Display Results
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Silent Dropout Score", round(score, 2))
    with col2:
        st.metric("Risk Level", risk_level)

    if risk_level == "HIGH":
        st.error("🚨 High silent dropout risk — immediate intervention required.")
    elif risk_level == "MEDIUM":
        st.warning("⚠️ Moderate risk — monitor closely.")
    else:
        st.success("✅ Low risk — patient engagement is healthy.")
