import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Silent Dropout Risk Prediction",
    page_icon="⚠️",
    layout="centered"
)

st.title("⚠️ Silent Dropout Risk Prediction")
st.write("Predict patient silent dropout risk using SVM")

# Load models (no cache decorator)
def load_models():
    model = pickle.load(open("svm_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
    return model, scaler, label_encoder, feature_columns

model, scaler, le, feature_columns = load_models()

# Numeric columns (must match what scaler was fitted on)
numeric_cols = [
    "Days_Since_Last_Contact",
    "Expected_Gap_Between_Visits_days",
    "Medicine_Refill_Delay_days",
    "Missed_Lab_Tests",
    "Days_Late_Follow_Up",
    "Silent_Dropout_Score"
]

st.subheader("🧾 Patient Details")
days_last_contact = st.number_input("Days Since Last Contact", 0, 365, 10)
expected_gap = st.number_input("Expected Gap Between Visits (days)", 0, 365, 7)
refill_delay = st.number_input("Medicine Refill Delay (days)", 0, 180, 2)
missed_tests = st.number_input("Missed Lab Tests", 0, 10, 0)
late_followup = st.number_input("Days Late for Follow-Up", 0, 365, 5)
silent_score = st.slider("Silent Dropout Score", 0.0, 10.0, 3.0)

if st.button("🔮 Predict Risk Level"):
    # Create input with ONLY numeric columns first
    input_data = {
        "Days_Since_Last_Contact": days_last_contact,
        "Expected_Gap_Between_Visits_days": expected_gap,
        "Medicine_Refill_Delay_days": refill_delay,
        "Missed_Lab_Tests": missed_tests,
        "Days_Late_Follow_Up": late_followup,
        "Silent_Dropout_Score": silent_score
    }
    input_df = pd.DataFrame([input_data])
    
    # Scale using the exact numeric columns the scaler expects
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Extend to full feature set expected by model (non-numeric get 0)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    prediction = model.predict(input_df)
    risk = le.inverse_transform(prediction)[0]
    
    if risk.lower() == "high":
        st.error("🚨 HIGH Risk of Silent Dropout")
    elif risk.lower() == "medium":
        st.warning("⚠️ MEDIUM Risk of Silent Dropout")
    else:
        st.success("✅ LOW Risk of Silent Dropout")
    
    st.write("**Predicted Risk Level:**", risk)
