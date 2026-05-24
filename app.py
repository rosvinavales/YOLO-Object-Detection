import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI 8.0: YOLO Project", layout="wide", page_icon="🤖")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["1. Live Object Detection", "2. Dataset & Training Results", "3. Comparative Analysis"])

# Load the model once
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

try:
    model = load_yolo()
except:
    st.error("Error: 'best.pt' not found. Please ensure your model weights are in the app folder.")

# ==========================================
# PAGE 1: LIVE OBJECT DETECTION
# ==========================================
if page == "1. Live Object Detection":
    st.title("🔍 Object Detection in Action")
    st.write("Upload a new image to test the model's ability to locate and classify objects.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col1:
            st.subheader("Input Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Detection Result")
            # Run YOLO Prediction
            img_array = np.array(image)
            results = model.predict(source=img_array, conf=0.25)
            
            # Plot results
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, use_column_width=True)
            st.success("Analysis Complete!")

# ==========================================
# PAGE 2: DATASET & TRAINING RESULTS
# ==========================================
elif page == "2. Dataset & Training Results":
    st.title("📊 Training Performance & Data")
    
    # Dataset Info
    st.info("**Dataset Source:** [Roboflow/Kaggle] | **Format:** YOLOv8 | **Size:** 5+ Documents/Images")

    # Metrics Row
    st.subheader("Final Recorded Metrics (Epoch 10)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", "0.951", "High Accuracy")
    m2.metric("Recall", "0.122", "Needs more Epochs")
    m3.metric("mAP50", "0.124", "Overall Grade")
    m4.metric("Train Loss", "1.917", "Decreasing")

    # The Table
    st.subheader("Performance Table")
    metrics_table = {
        "Metric": ["Precision", "Recall", "mAP50", "Training Box Loss", "Validation Box Loss"],
        "Value": ["0.9512", "0.1218", "0.1242", "1.9170", "1.6854"],
        "Status": ["Excellent", "Low (Underfit)", "Stable", "Good", "Good"]
    }
    st.table(pd.DataFrame(metrics_table))

    # The Graph
    st.subheader("Training Progress Visual")
    if os.path.exists("results.png"):
        st.image("results.png", caption="Loss and Metric Graphs over 10 Epochs", use_column_width=True)
    else:
        st.warning("Upload 'results.png' to see the progress graphs.")

# ==========================================
# PAGE 3: COMPARATIVE ANALYSIS
# ==========================================
elif page == "3. Comparative Analysis":
    st.title("📖 Comparative Analysis Questions")
    st.write("Answers based on the YOLO framework and training experiment.")

    analysis_data = [
        {"Question": "1. What is object detection?", "Answer": "A computer vision task that identifies both WHAT an object is (classification) and WHERE it is located (localization) using bounding boxes."},
        {"Question": "2. How does YOLO perform object detection?", "Answer": "YOLO analyzes the entire image in a single pass. It predicts object classes and bounding boxes simultaneously, making it incredibly fast for real-time use."},
        {"Question": "3. What is the role of a pre-annotated dataset?", "Answer": "It provides 'ground truth' examples. Since images are already marked with boxes, the AI can learn by comparing its guesses to the correct labels provided by humans."},
        {"Question": "4. What do Precision, Recall, and mAP measure?", "Answer": "Precision: Accuracy of hits. Recall: Ability to find all objects. mAP: Combined score of precision and recall performance."},
        {"Question": "5. What challenges did you encounter?", "Answer": "Key challenges included low recall due to limited training epochs (10) and the requirement for high-power T4 GPUs to process matrix calculations efficiently."},
        {"Question": "6. How can performance be improved?", "Answer": "By increasing training epochs (to 50 or 100), using a larger dataset, and applying data augmentation (flipping/rotating images)."}
    ]

    for item in analysis_data:
        with st.expander(item["Question"]):
            st.write(item["Answer"])

    st.markdown("---")
    st.write("**Submission Deadline:** May 25, 2026")
    st.write("**Status:** ✅ All Requirements Met")
