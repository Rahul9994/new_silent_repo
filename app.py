import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Silent Dropout Risk Prediction",
    page_icon="⚠️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1e3a8a; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {text-align: center; color: #6b7280; margin-bottom: 2rem;}
    .risk-card {padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;}
    .high-risk {background-color: #fee2e2; border-left: 4px solid #dc2626;}
    .medium-risk {background-color: #fef3c7; border-left: 4px solid #d97706;}
    .low-risk {background-color: #dcfce7; border-left: 4px solid #16a34a;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚠️ Silent Dropout Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered patient retention risk assessment</p>', unsafe_allow_html=True)

def load_models_safely():
    try:
        model_files = ["svm_model.pkl", "scaler.pkl", "label_encoder.pkl", "feature_columns.pkl"]
        for filepath in model_files:
            if not os.path.exists(filepath):
                st.error(f"❌ Missing: {filepath}")
                return None, None, None, None
        
        model = pickle.load(open("svm_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
        feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
        
        return model, scaler, label_encoder, feature_columns
    except Exception as e:
        st.error(f"❌ Load error: {str(e)}")
        return None, None, None, None

@st.cache_data
def get_feature_info():
    return {
        "Days_Since_Last_Contact": {"min": 0, "max": 365, "default": 10},
        "Expected_Gap_Between_Visits_days": {"min": 0, "max": 365, "default": 7},
        "Medicine_Refill_Delay_days": {"min": 0, "max": 180, "default": 2},
        "Missed_Lab_Tests": {"min": 0, "max": 10, "default": 0},
        "Days_Late_Follow_Up": {"min": 0, "max": 365, "default": 5},
        "Silent_Dropout_Score": {"min": 0.0, "max": 10.0, "default": 3.0}
    }

def create_risk_visualization(risk_level, confidence=None):
    if risk_level.lower() == "high":
        color, icon, bg = "#dc2626", "🔴", "#fee2e2"
    elif risk_level.lower() == "medium":
        color, icon, bg = "#d97706", "🟡", "#fef3c7"
    else:
        color, icon, bg = "#16a34a", "🟢", "#dcfce7"
    
    html = f'<div style="background-color: {bg}; border-left: 4px solid {color}; padding: 1.5rem; border-radius: 0.5rem;">'
    html += f'<h3 style="color: {color}; margin: 0;">{icon} {risk_level.upper()}</h3>'
    if confidence: html += f'<p>Confidence: {confidence:.1%}</p>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# Load models
model, scaler, le, feature_columns = load_models_safely()
if model is None: st.stop()

# Sidebar info
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.info("**SVM for patient dropout prediction**\nRisk: Low 🟢 | Medium 🟡 | High 🔴")

feature_info = get_feature_info()

# Inputs with session state
if 'input_values' not in st.session_state:
    st.session_state.input_values = {k: v['default'] for k, v in feature_info.items()}

st.header("🧾 Patient Data")
col1, col2, col3 = st.columns(3)
if col1.button("🟢 Low Risk Test"): 
    st.session_state.input_values = {k: v['min'] for k, v in feature_info.items()}
if col2.button("🟡 Medium Test"):
    vals = {k: v['min'] for k, v in feature_info.items()}
    vals.update({"Days_Since_Last_Contact": 30, "Silent_Dropout_Score": 5.0})
    st.session_state.input_values = vals
if col3.button("🔴 High Test"):
    vals = {k: v['max'] for k, v in feature_info.items()}
    st.session_state.input_values = vals

input_values = st.session_state.input_values
col1, col2 = st.columns(2)

with col1:
    input_values['Days_Since_Last_Contact'] = st.number_input("Days Since Contact", **feature_info['Days_Since_Last_Contact'], key="dsc")
    input_values['Expected_Gap_Between_Visits_days'] = st.number_input("Expected Gap (days)", **feature_info['Expected_Gap_Between_Visits_days'], key="eg")
    input_values['Days_Late_Follow_Up'] = st.number_input("Days Late Follow-up", **feature_info['Days_Late_Follow_Up'], key="dlf")

with col2:
    input_values['Medicine_Refill_Delay_days'] = st.number_input("Refill Delay (days)", **feature_info['Medicine_Refill_Delay_days'], key="mrd")
    input_values['Missed_Lab_Tests'] = st.number_input("Missed Tests", **feature_info['Missed_Lab_Tests'], key="mlt")
    input_values['Silent_Dropout_Score'] = st.slider("Dropout Score", **{k: float(v) for k, v in feature_info['Silent_Dropout_Score'].items()}, key="sds")

st.session_state.input_values = input_values

if st.button("🔮 Predict Risk", use_container_width=True):
    try:
        with st.spinner("Analyzing..."):
            # CRITICAL FIX: Get EXACT scaler expected features
            scaler_features = getattr(scaler, 'feature_names_in_', None)
            
            with st.sidebar:
                st.markdown("### 🔧 Debug")
                st.write("**Scaler expects:**", scaler_features)
                st.write("**Model classes:**", model.classes_)
            
            # Create full DataFrame with ALL feature_columns first (0s for missing)
            input_df = pd.DataFrame([input_values]).reindex(columns=feature_columns, fill_value=0.0)
            
            # FIXED: Scale ONLY scaler's expected columns, in EXACT order
            if scaler_features is not None:
                # Select ONLY matching columns in EXACT order scaler expects
                scale_cols = [col for col in scaler_features if col in input_df.columns]
                input_df[scale_cols] = scaler.transform(input_df[scale_cols])
            else:
                # Fallback: scale all numeric
                numeric_cols = input_df.select_dtypes(include=[np.number]).columns
                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            
            with st.sidebar:
                st.write("**Final input to model:**")
                st.json({k: float(v) for k, v in input_df.iloc[0].items() if abs(v) > 1e-10})
            
            # Predict
            pred = model.predict(input_df)[0]
            risk_level = le.inverse_transform([pred])[0]
            
            conf = None
            try: conf = np.max(model.predict_proba(input_df)[0])
            except: pass

        st.markdown("---")
        st.header("📊 Results")
        col1, col2 = st.columns([2,1])
        with col1: create_risk_visualization(risk_level, conf)
        with col2:
            score = sum([input_values[k] * (0.2 if k=='Days_Since_Last_Contact' else 0.3 if k=='Days_Late_Follow_Up' else 0.25 if k=='Medicine_Refill_Delay_days' else 5 if k=='Missed_Lab_Tests' else 2) for k in input_values])
            st.metric("Risk Score", f"{score:.1f}/100")
            if conf: st.metric("Confidence", f"{conf:.1%}")
            st.metric("Date", datetime.now().strftime("%Y-%m-%d %H:%M"))

        st.markdown("### 💡 Action Items")
        if "high" in risk_level.lower():
            st.error("**URGENT:** Contact within 24h, urgent follow-up")
        elif "medium" in risk_level.lower():
            st.warning("**PROACTIVE:** Contact within 3 days, reminders")
        else:
            st.success("**MONITOR:** Continue standard care")

        # Chart
        fig, ax = plt.subplots(figsize=(10,5))
        feats, vals = list(input_values.keys()), list(input_values.values())
        norms = [(v / feature_info[f]['max']) * 100 for f,v in zip(feats,vals)]
        colors = ['green' if v<33 else 'orange' if v<66 else 'red' for v in norms]
        ax.barh(feats, norms, color=colors)
        ax.set_xlabel('Risk %'); ax.set_title('Risk Factors')
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Check sidebar debug. Retrain scaler on exact input features if mismatch persists.")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#6b7280;padding:2rem;">Silent Dropout Prediction | ML-Powered Healthcare</div>', unsafe_allow_html=True)
