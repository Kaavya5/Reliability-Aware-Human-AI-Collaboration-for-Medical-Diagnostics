import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import torch
import cv2

# Import custom modules
from model import MCDropoutResNet, predict_with_uncertainty
from utils import get_transforms, calculate_risk_metrics, determine_priority

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Risk-Aware Clinical AI Dashboard", layout="wide", page_icon="🏥")

# Custom CSS for modern medical UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    .alert-green {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        text-align: center;
        font-weight: bold;
    }
    .alert-orange {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ffeeba;
        text-align: center;
        font-weight: bold;
    }
    .alert-red {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        text-align: center;
        font-weight: bold;
    }
    .disclaimer {
        font-size: 0.9rem;
        color: #e74c3c;
        font-style: italic;
        text-align: center;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_pytorch_model():
    """Loads the ResNet50 model equipped with MC Dropout."""
    model = MCDropoutResNet(num_classes=1, dropout_prob=0.5)
    
    # Load the fine-tuned reality checkpoint
    try:
        model.load_state_dict(torch.load("demo_checkpoint.pth", map_location='cpu'))
    except FileNotFoundError:
        pass # Fallback to random if not fetched yet
    
    model.eval() # Base state. MC Dropout prediction sets it to train automatically.
    return model

# --- IMAGE OVERLAY UTILITY ---
def get_gradcam_overlay(image_pil, cam):
    """Overlays the Grad-CAM heatmap on the original PIL image."""
    img_array = np.array(image_pil.resize((224, 224)))
    # Apply colormap to CAM
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    # Overlay loosely
    overlay = cv2.addWeighted(img_array, 0.6, cam_heatmap, 0.4, 0)
    return Image.fromarray(overlay)

# Load global models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pytorch_model = load_pytorch_model().to(device)
transforms = get_transforms()

# --- DASHBOARD LOGIC ---
st.title("🏥 Reliability-Aware Human-in-the-Loop Clinical AI")
st.markdown("A demonstration of collaborative AI utilizing PyTorch & Monte Carlo Dropout for real-time uncertainty estimation.")

# Create the top layout: Left Panel (Input) and Center Panel (Output)
col_input, col_output = st.columns([1, 2])

with col_input:
    st.header("1. Input Panel")
    with st.container(border=True):
        input_method = st.radio("Image Source", ["Use Sample Image", "Upload Custom X-ray"], horizontal=True)
        
        uploaded_file = None
        sample_image_path = None
        
        if input_method == "Upload Custom X-ray":
            uploaded_file = st.file_uploader("Upload Medical Image (X-ray/Scan)", type=["png", "jpg", "jpeg"])
        else:
            sample_choice = st.selectbox("Select a Sample Patient", 
                                         ["NORMAL_1.jpeg", "NORMAL_2.jpeg", "PNEUMONIA_1.jpeg", "PNEUMONIA_2.jpeg"])
            sample_image_path = f"data/samples/{sample_choice}"
        
        st.markdown("---")
        st.subheader("Clinical Parameters")
        severity = st.slider("Severity Level (S)", min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                             help="Clinician's initial assessment of potential severity (1=Low, 5=High)")
        safety_weight = st.slider("Safety Weight (W)", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                  help="Weight placed on model uncertainty when calculating risk (0=Ignore, 1=High penalty)")
        
        run_analysis = st.button("Run Inference", type="primary", use_container_width=True)
        
        st.markdown("---")
        workflow_toggle = st.radio("Show Dashboard Mode", ["Collaborative Workflow", "Traditional Workflow"], 
                                   help="Toggle to see how the system behaves under different paradigms.")

with col_output:
    st.header("2. AI Output Panel")
    
    if (uploaded_file is not None or sample_image_path is not None) and run_analysis:
        try:
            if input_method == "Upload Custom X-ray":
                image_pil = Image.open(uploaded_file).convert('RGB')
            else:
                image_pil = Image.open(sample_image_path).convert('RGB')
        except Exception:
            st.error("Invalid image format or sample not found. Ensure `fetch_samples.py` was run.")
            st.stop()
            
        with st.spinner("Running PyTorch Monte Carlo Inference..."):
            # Image preprocessing
            input_tensor = transforms(image_pil).unsqueeze(0).to(device)
            
            # CORE LOGIC: Get P, C, U via PyTorch MC Dropout Inference
            p, c, u = predict_with_uncertainty(pytorch_model, input_tensor, num_samples=20)
            
            # Get GradCAM
            cam, _ = pytorch_model.generate_gradcam(input_tensor)
            cam_overlay = get_gradcam_overlay(image_pil, cam)
            
            # Calculate Risk Engine rules
            risk, reliability, adjusted_risk = calculate_risk_metrics(p, c, u, severity, safety_weight)
            
            # Get Priority classification
            t1_urgent = 3.5
            t2_review = 2.0
            priority = determine_priority(adjusted_risk, t1_urgent, t2_review)
            
            # --- DISPLAY IMAGES ---
            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.image(image_pil, caption="Original Uploaded X-ray", use_container_width=True)
            with img_c2:
                st.image(cam_overlay, caption="Grad-CAM Focus Heatmap", use_container_width=True)
            
            # --- MOCK CLASSIFICATION LABEL (since no true weights map yet) ---
            label = "Positive (Abnormality Detected)" if p > 0.5 else "Negative (Normal)"
            st.markdown(f"**Predicted Phenotype:** {label}")

            # Layout the metrics
            st.markdown("### Model Outputs")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-label" title="Prediction Probability">Probability (P)</div><div class="metric-value">{p:.3f}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-label" title="Model Confidence">Confidence (C)</div><div class="metric-value">{c:.3f}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-label" title="Model Epistemic Uncertainty">Uncertainty (U)</div><div class="metric-value">{u:.3f}</div></div>', unsafe_allow_html=True)
                
            st.markdown("### Risk Assessment")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Base Risk (P × S)</div><div class="metric-value">{risk:.2f}</div></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Reliability (C × (1-U))</div><div class="metric-value">{reliability:.2f}</div></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="metric-card"><div class="metric-label" title="Risk + (1-Reliability) × W">Adjusted Risk</div><div class="metric-value">{adjusted_risk:.2f}</div></div>', unsafe_allow_html=True)
            
            # DECISION LOGIC
            st.markdown("### Priority Level")
            
            if priority == "Urgent":
                st.markdown('<div class="alert-red">🚨 URGENT ATTENTION REQUIRED</div>', unsafe_allow_html=True)
                reasoning = f"Adjusted Risk ({adjusted_risk:.2f}) exceeds Urgent threshold ({t1_urgent}). The interplay of potential disease severity ({severity}) and model uncertainty ({u:.2f}) forces an immediate review."
            elif priority == "Review":
                st.markdown('<div class="alert-orange">⚠️ NEEDS REVIEW</div>', unsafe_allow_html=True)
                reasoning = f"Adjusted Risk ({adjusted_risk:.2f}) exceeds Review threshold ({t2_review}). Consider verifying AI focal points on Grad-CAM."
            else:
                st.markdown('<div class="alert-green">✅ ROUTINE (SAFE)</div>', unsafe_allow_html=True)
                reasoning = f"Adjusted Risk ({adjusted_risk:.2f}) is below Review threshold ({t2_review}). Case is safe for routine backlog."
            
            st.info(f"**Explanation:** {reasoning}")
            
            st.markdown('<div class="disclaimer">⚠️ AI suggests priority — final diagnosis and treatment decision must be made by a qualified clinician.</div>', unsafe_allow_html=True)

    elif uploaded_file is None and sample_image_path is None:
        st.info("Please select or upload a chest X-ray image and click 'Run Inference' to evaluate.")

st.markdown("---")

# --- ANALYTICS DASHBOARD ---
st.header("3. Analytics Dashboard")
st.markdown("Comparing Cohort Performance: Traditional AI (Autonomous) vs Collaborative AI (Human-in-the-Loop)")

# Mock Data Generation for Cohort
@st.cache_data
def get_cohort_data():
    np.random.seed(42)  # Maintain stable metrics
    n = 1000
    
    # Generate ground truth
    actual_positive = np.random.choice([0, 1], p=[0.8, 0.2], size=n)
    
    # Generate AI probabilities that are largely accurate (e.g. ~85% accuracy)
    # If positive, skew probabilities towards 1.0. If negative, skew towards 0.0
    ai_prob = np.where(
        actual_positive == 1,
        np.random.beta(a=8, b=2, size=n),  # Mean ~0.8
        np.random.beta(a=2, b=8, size=n)   # Mean ~0.2
    )
    
    return pd.DataFrame({
        "Patient_ID": range(n),
        "Actual_Positive": actual_positive, 
        "AI_Prob": ai_prob,
        "AI_Uncertainty": np.random.uniform(0, 0.5, size=n), # Keep uncertainty random for demo spread
        "Severity": np.random.uniform(1, 5, size=n)
    })

data = get_cohort_data()

# Simulation Logic
# Traditional: Fixed threshold (e.g., Prob > 0.5)
data['Trad_Pred'] = (data['AI_Prob'] > 0.5).astype(int)
trad_missed = len(data[(data['Actual_Positive'] == 1) & (data['Trad_Pred'] == 0)])
trad_false_alerts = len(data[(data['Actual_Positive'] == 0) & (data['Trad_Pred'] == 1)])
trad_accuracy = (data['Trad_Pred'] == data['Actual_Positive']).mean()
trad_review_load = len(data[data['Trad_Pred'] == 1]) 

# Collaborative: Risk aware
data['Collab_Risk'] = data['AI_Prob'] * data['Severity']
data['Collab_Confidence'] = 0.9 # assumed stable baseline
data['Collab_Reliability'] = data['Collab_Confidence'] * (1 - data['AI_Uncertainty']) 
data['Collab_Adjusted_Risk'] = data['Collab_Risk'] + (1 - data['Collab_Reliability']) * 0.5

# Collaborative thresholds
T1, T2 = 3.5, 2.0
data['Collab_Priority'] = 'Routine'
data['Collab_Priority'] = np.where(data['Collab_Adjusted_Risk'] >= T2, 'Review', data['Collab_Priority'])
data['Collab_Priority'] = np.where(data['Collab_Adjusted_Risk'] >= T1, 'Urgent', data['Collab_Priority'])

collab_missed = len(data[(data['Actual_Positive'] == 1) & (data['Collab_Priority'] == 'Routine')])
collab_false_alerts = len(data[(data['Actual_Positive'] == 0) & (data['Collab_Priority'] == 'Urgent')])
collab_accuracy = trad_accuracy 
collab_review_load = len(data[data['Collab_Priority'] != 'Routine'])

missed_reduction = ((trad_missed - collab_missed) / trad_missed) * 100 if trad_missed else 0
alerts_reduction = ((trad_false_alerts - collab_false_alerts) / trad_false_alerts) * 100 if trad_false_alerts else 0
efficiency_gain = ((trad_review_load - collab_review_load) / trad_review_load) * 100 if trad_review_load else 0

# Visuals
d1, d2, d3, d4 = st.columns(4)

if workflow_toggle == "Traditional Workflow":
    d1.metric(label="Model Accuracy", value=f"{trad_accuracy*100:.1f}%")
    d2.metric(label="Missed Critical Cases", value=trad_missed)
    d3.metric(label="False Urgent Alerts", value=trad_false_alerts)
    d4.metric(label="Cases to Review", value=trad_review_load)
else:
    d1.metric(label="Model Accuracy", value=f"{collab_accuracy*100:.1f}%", help="Accuracy of base AI is identical")
    d2.metric(label="Missed Critical Cases", value=collab_missed, delta=f"-{missed_reduction:.1f}%", delta_color="inverse")
    d3.metric(label="False Urgent Alerts", value=collab_false_alerts, delta=f"-{alerts_reduction:.1f}%", delta_color="inverse")
    d4.metric(label="Cases to Review", value=collab_review_load, delta=f"{efficiency_gain:.1f}% (Efficiency Gain)", delta_color="normal")


st.markdown("### Workflows Comparison Charts")
c1, c2 = st.columns(2)

with c1:
    fig1 = go.Figure(data=[
        go.Bar(name='Traditional AI', x=['Missed Cases', 'False Alerts'], y=[trad_missed, trad_false_alerts], marker_color='#34495e'),
        go.Bar(name='Collaborative AI', x=['Missed Cases', 'False Alerts'], y=[collab_missed, collab_false_alerts], marker_color='#3498db')
    ])
    fig1.update_layout(barmode='group', title="Safety Metrics (Lower is Better)")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    fig2 = go.Figure(data=[
        go.Bar(name='Routine (No Review)', x=['Traditional AI', 'Collaborative AI'], y=[len(data) - trad_review_load, len(data[data['Collab_Priority'] == 'Routine'])], marker_color='#2ecc71'),
        go.Bar(name='Review / Flagged', x=['Traditional AI', 'Collaborative AI'], y=[trad_review_load, collab_review_load], marker_color='#f39c12')
    ])
    fig2.update_layout(barmode='stack', title="Workload Distribution (Efficiency)")
    st.plotly_chart(fig2, use_container_width=True)
