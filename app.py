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

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #fee2e2;
        border-left: 4px solid #dc2626;
    }
    .medium-risk {
        background-color: #fef3c7;
        border-left: 4px solid #d97706;
    }
    .low-risk {
        background-color: #dcfce7;
        border-left: 4px solid #16a34a;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">⚠️ Silent Dropout Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered patient retention risk assessment using machine learning</p>', unsafe_allow_html=True)

def load_models_safely():
    """Load models with proper error handling"""
    try:
        model_files = {
            "model": "svm_model.pkl",
            "scaler": "scaler.pkl", 
            "label_encoder": "label_encoder.pkl",
            "feature_columns": "feature_columns.pkl"
        }
        
        for name, filepath in model_files.items():
            if not os.path.exists(filepath):
                st.error(f"❌ Missing file: {filepath}")
                st.info("Please ensure all model files are in the same directory as the app.")
                return None, None, None, None
        
        model = pickle.load(open("svm_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
        feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
        
        return model, scaler, label_encoder, feature_columns
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None

@st.cache_data
def get_feature_info():
    """Get information about features for better UX"""
    return {
        "Days_Since_Last_Contact": {
            "description": "Days since the patient was last contacted",
            "min": 0, "max": 365, "default": 10,
            "help": "Higher values indicate longer gaps in communication"
        },
        "Expected_Gap_Between_Visits_days": {
            "description": "Expected time between patient visits",
            "min": 0, "max": 365, "default": 7,
            "help": "Based on treatment protocol and patient condition"
        },
        "Medicine_Refill_Delay_days": {
            "description": "Delay in medicine refill",
            "min": 0, "max": 180, "default": 2,
            "help": "Indicates medication adherence issues"
        },
        "Missed_Lab_Tests": {
            "description": "Number of missed laboratory tests",
            "min": 0, "max": 10, "default": 0,
            "help": "Missed tests may indicate disengagement"
        },
        "Days_Late_Follow_Up": {
            "description": "Days late for scheduled follow-up",
            "min": 0, "max": 365, "default": 5,
            "help": "Delays in follow-up appointments are red flags"
        },
        "Silent_Dropout_Score": {
            "description": "Preliminary silent dropout risk score",
            "min": 0.0, "max": 10.0, "default": 3.0,
            "help": "Composite score from previous assessments"
        }
    }

def create_risk_visualization(risk_level, confidence_score=None):
    """Create visual risk indicators"""
    if risk_level.lower() == "high":
        color = "#dc2626"
        icon = "🔴"
        bg_color = "#fee2e2"
    elif risk_level.lower() == "medium":
        color = "#d97706"
        icon = "🟡"
        bg_color = "#fef3c7"
    else:
        color = "#16a34a"
        icon = "🟢"
        bg_color = "#dcfce7"
    
    st.markdown(f"""
    <div class="risk-card" style="background-color: {bg_color}; border-left: 4px solid {color};">
        <h3 style="color: {color}; margin: 0;">{icon} Risk Level: {risk_level.upper()}</h3>
        {f'<p>Confidence: {confidence_score:.1%}</p>' if confidence_score else ''}
    </div>
    """, unsafe_allow_html=True)

def show_model_info():
    """Display model information and debug info in sidebar"""
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.info("""
        **Model Type:** Support Vector Machine (SVM)
        
        **Purpose:** Predict patient silent dropout risk
        
        **Risk Levels:**
        - 🟢 Low Risk
        - 🟡 Medium Risk  
        - 🔴 High Risk
        
        **Features Used:**
        - Contact history
        - Visit patterns
        - Medication adherence
        - Lab test compliance
        - Follow-up behavior
        """)
        
        st.header("📊 Feature Importance")
        st.write("The model considers multiple factors:")
        st.write("1. **Communication gaps** (Days since last contact)")
        st.write("2. **Visit adherence** (Expected vs actual gaps)")
        st.write("3. **Medication compliance** (Refill delays)")
        st.write("4. **Test compliance** (Missed lab tests)")
        st.write("5. **Follow-up patterns** (Days late)")
        st.write("6. **Historical risk** (Silent dropout score)")

# Load models
model, scaler, le, feature_columns = load_models_safely()

if model is None:
    st.stop()

# Show model info
show_model_info()

# Get feature information
feature_info = get_feature_info()

# Main input section
st.header("🧾 Patient Assessment Form")
st.markdown("---")

# Quick test buttons for debugging
col1, col2, col3 = st.columns(3)
if col1.button("🟢 Test Low Risk (All 0)"):
    for key in input_values:
        st.session_state[key] = feature_info[key]['min']
if col2.button("🟡 Test Medium Risk"):
    st.session_state['Days_Since_Last_Contact'] = 30
    st.session_state['Medicine_Refill_Delay_days'] = 10
    st.session_state['Silent_Dropout_Score'] = 5.0
if col3.button("🔴 Test High Risk"):
    st.session_state['Days_Since_Last_Contact'] = 90
    st.session_state['Days_Late_Follow_Up'] = 45
    st.session_state['Silent_Dropout_Score'] = 8.0

# Create two columns for inputs
col1, col2 = st.columns(2)

# Use session_state for input persistence
if 'input_values' not in st.session_state:
    st.session_state.input_values = {}

input_values = st.session_state.input_values

with col1:
    st.subheader("📞 Communication & Visits")
    
    input_values['Days_Since_Last_Contact'] = st.number_input(
        "Days Since Last Contact",
        min_value=feature_info['Days_Since_Last_Contact']['min'],
        max_value=feature_info['Days_Since_Last_Contact']['max'],
        value=feature_info['Days_Since_Last_Contact']['default'],
        key='Days_Since_Last_Contact',
        help=feature_info['Days_Since_Last_Contact']['help']
    )
    
    input_values['Expected_Gap_Between_Visits_days'] = st.number_input(
        "Expected Gap Between Visits (days)",
        min_value=feature_info['Expected_Gap_Between_Visits_days']['min'],
        max_value=feature_info['Expected_Gap_Between_Visits_days']['max'],
        value=feature_info['Expected_Gap_Between_Visits_days']['default'],
        key='Expected_Gap_Between_Visits_days',
        help=feature_info['Expected_Gap_Between_Visits_days']['help']
    )
    
    input_values['Days_Late_Follow_Up'] = st.number_input(
        "Days Late for Follow-Up",
        min_value=feature_info['Days_Late_Follow_Up']['min'],
        max_value=feature_info['Days_Late_Follow_Up']['max'],
        value=feature_info['Days_Late_Follow_Up']['default'],
        key='Days_Late_Follow_Up',
        help=feature_info['Days_Late_Follow_Up']['help']
    )

with col2:
    st.subheader("💊 Treatment Compliance")
    
    input_values['Medicine_Refill_Delay_days'] = st.number_input(
        "Medicine Refill Delay (days)",
        min_value=feature_info['Medicine_Refill_Delay_days']['min'],
        max_value=feature_info['Medicine_Refill_Delay_days']['max'],
        value=feature_info['Medicine_Refill_Delay_days']['default'],
        key='Medicine_Refill_Delay_days',
        help=feature_info['Medicine_Refill_Delay_days']['help']
    )
    
    input_values['Missed_Lab_Tests'] = st.number_input(
        "Missed Lab Tests",
        min_value=feature_info['Missed_Lab_Tests']['min'],
        max_value=feature_info['Missed_Lab_Tests']['max'],
        value=feature_info['Missed_Lab_Tests']['default'],
        key='Missed_Lab_Tests',
        help=feature_info['Missed_Lab_Tests']['help']
    )
    
    input_values['Silent_Dropout_Score'] = st.slider(
        "Silent Dropout Score",
        min_value=float(feature_info['Silent_Dropout_Score']['min']),
        max_value=float(feature_info['Silent_Dropout_Score']['max']),
        value=float(feature_info['Silent_Dropout_Score']['default']),
        step=0.1,
        key='Silent_Dropout_Score',
        help=feature_info['Silent_Dropout_Score']['help']
    )

# Update session_state
st.session_state.input_values = input_values

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Risk Level", use_container_width=True, type="primary")

if predict_button:
    try:
        with st.spinner("🤖 Analyzing patient data..."):
            # Debug info to sidebar
            with st.sidebar:
                st.markdown("### 🔧 Debug Info")
                st.write("**Model classes:**", model.classes_)
                if hasattr(scaler, 'feature_names_in_'):
                    st.write("**Scaler features:**", list(scaler.feature_names_in_))
                st.write("**Expected features:**", list(feature_columns))
            
            # FIXED: Create input ONLY with 6 known numeric features FIRST
            input_df = pd.DataFrame([input_values])
            
            # FIXED: Scale EXACTLY the 6 input columns (matches training)
            numeric_cols = list(input_values.keys())
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            
            # FIXED: THEN reindex to full feature set (adds missing as 0.0)
            input_df = input_df.reindex(columns=feature_columns, fill_value=0.0)
            
            # Debug: Show processed input
            with st.sidebar:
                st.write("**Scaled input:**")
                st.json(input_df.iloc[0].to_dict())
            
            # Predict
            prediction = model.predict(input_df)
            risk_level = le.inverse_transform(prediction)[0]
            
            # Confidence if available
            confidence = None
            try:
                probabilities = model.predict_proba(input_df)
                confidence = np.max(probabilities[0])
            except (AttributeError, ValueError):
                pass
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            create_risk_visualization(risk_level, confidence)
        
        with result_col2:
            st.markdown("### 📈 Risk Metrics")
            
            total_risk_score = (
                input_values['Days_Since_Last_Contact'] * 0.2 +
                input_values['Days_Late_Follow_Up'] * 0.3 +
                input_values['Medicine_Refill_Delay_days'] * 0.25 +
                input_values['Missed_Lab_Tests'] * 5 +
                input_values['Silent_Dropout_Score'] * 2
            )
            
            st.metric("Risk Score", f"{total_risk_score:.1f}/100")
            
            if confidence:
                st.metric("Model Confidence", f"{confidence:.1%}")
            
            st.metric("Assessment Date", datetime.now().strftime("%Y-%m-%d"))
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if risk_level.lower() == "high":
            st.error("""
            **Immediate Action Required:**
            - Contact patient within 24 hours
            - Schedule urgent follow-up appointment
            - Review treatment plan and barriers to care
            - Consider case management intervention
            """)
        elif risk_level.lower() == "medium":
            st.warning("""
            **Proactive Measures Recommended:**
            - Contact patient within 3 days
            - Reinforce importance of follow-up care
            - Address any identified barriers
            - Schedule reminder calls/messages
            """)
        else:
            st.success("""
            **Continue Standard Care:**
            - Maintain regular follow-up schedule
            - Monitor for any changes in behavior
            - Provide patient education as needed
            - Document all interactions
            """)
        
        # Feature visualization
        st.markdown("### 🔍 Contributing Factors")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        features = list(input_values.keys())
        values = list(input_values.values())
        
        normalized_values = []
        for i, (feature, value) in enumerate(zip(features, values)):
            max_val = feature_info.get(feature, {}).get('max', 10)
            normalized_values.append((value / max_val) * 100)
        
        colors = ['#3b82f6' if v < 33 else '#f59e0b' if v < 66 else '#ef4444' for v in normalized_values]
        
        bars = ax.barh(features, normalized_values, color=colors)
        ax.set_xlabel('Risk Contribution (%)')
        ax.set_title('Patient Risk Factor Analysis')
        ax.set_xlim(0, 100)
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{value}', ha='left', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Check sidebar debug info and model files.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>Silent Dropout Risk Prediction System</strong></p>
    <p>Powered by Machine Learning | Built for Healthcare Excellence</p>
</div>
""", unsafe_allow_html=True)
