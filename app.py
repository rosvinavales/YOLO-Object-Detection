import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI 8.0: YOLO Activity", layout="wide")

# --- LOAD MODEL ---
# This fulfills Requirement 2: Use a pretrained/trained YOLO model
@st.cache_resource
def load_yolo_model():
    # If using GitHub, make sure 'best.pt' is in the root folder
    return YOLO("best.pt")

# --- NAVIGATION ---
st.sidebar.title("Activity Requirements")
menu = ["Project Overview", "1. Dataset Prep", "2. Training Results", "3. Image Testing", "4. Comparative Analysis"]
choice = st.sidebar.radio("Select Section:", menu)

# ==========================================
# SECTION: DATASET PREPARATION
# ==========================================
if choice == "1. Dataset Prep":
    st.title("📂 Dataset Preparation")
    st.write("### Dataset Link (Requirement 5)")
    st.success("Dataset Link: [PASTE YOUR ROBOFLOW/KAGGLE LINK HERE]")
    
    st.write("### Organization")
    st.markdown("""
    As per instructions, the dataset was organized into:
    - **Training set:** For model learning.
    - **Validation set:** For tuning.
    - **Testing set:** For final evaluation.
    - **Config:** `data.yaml` was used to map class labels.
    """)

# ==========================================
# SECTION: TRAINING RESULTS & METRICS
# ==========================================
elif choice == "2. Training Results":
    st.title("📊 Training Results & Metrics")
    st.write("Requirement 2: Record specific metrics during training.")
    
    # These are the values from your CSV
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", "0.951")
    col2.metric("Recall", "0.122")
    col3.metric("mAP50", "0.124")
    col4.metric("Train Loss", "1.917")

    st.write("### Performance Screenshot (Requirement 5)")
    if os.path.exists("results.png"):
        st.image("results.png", caption="Training Results Graph")
    else:
        st.warning("Please upload 'results.png' to your GitHub repo to show this.")

# ==========================================
# SECTION: IMAGE TESTING
# ==========================================
elif choice == "3. Image Testing":
    st.title("🖼️ Image Testing (Requirement 3)")
    st.write("Test the model on unseen images with bounding boxes, labels, and confidence.")
    
    uploaded_file = st.file_uploader("Upload Test Image", type=['jpg','png','jpeg'])
    
    if uploaded_file:
        model = load_yolo_model()
        image = Image.open(uploaded_file)
        
        # Run Detection
        results = model.predict(source=image, conf=0.25)
        
        # Display Results
        # Requirement 3: display bounding boxes, class labels, and confidence scores
        res_plotted = results[0].plot() 
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption="Detection Output", use_column_width=True)

# ==========================================
# SECTION: COMPARATIVE ANALYSIS
# ==========================================
elif choice == "4. Comparative Analysis":
    st.title("🧠 Comparative Analysis (Requirement 4)")
    st.write("After training and testing the model, here are the answers to the analysis questions:")
    
    qa = {
        "1. What is object detection?": "Object detection is a computer vision task that determines both what the object is (classification) and where it is located (localization) using bounding boxes.",
        "2. How does YOLO perform object detection?": "YOLO (You Only Look Once) analyzes the entire image in a single pass through a neural network, predicting bounding boxes and class probabilities simultaneously.",
        "3. What is the role of a pre-annotated dataset?": "It provides ground-truth labels and boxes so the model can learn from correct examples without manual human labeling during training.",
        "4. What do Precision, Recall, and mAP measure?": "Precision measures prediction accuracy. Recall measures detection coverage. mAP (mean Average Precision) is the overall performance score.",
        "5. What challenges did you encounter?": "Managing hardware acceleration (T4 GPU) and balancing training epochs to improve low recall scores.",
        "6. How can object detection performance be improved?": "By increasing the dataset size, using data augmentation, and training for more epochs (e.g., 50 or 100)."
    }
    
    for q, a in qa.items():
        with st.expander(q):
            st.write(a)
