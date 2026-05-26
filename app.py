import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI 8.0: Corn Leaf Disease Detection",
    page_icon="🌽",
    layout="wide"
)

# --- LOAD TRAINED MODEL ---
@st.cache_resource
def load_yolo_model():
    # Requirement 2: Build model using trained 'best.pt' weights
    return YOLO("best.pt")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("YOLO Activity Project")
menu = [
    "1. Dataset Preparation", 
    "2. Training Results & Metrics", 
    "3. Image Testing & Outputs", 
    "4. Comparative Analysis",
    "5. Project Source Code"
]
choice = st.sidebar.radio("Navigate to:", menu)

# ==========================================
# SECTION 1: DATASET PREPARATION
# ==========================================
if choice == "1. Dataset Preparation":
    st.title("📂 Dataset Preparation")
    st.write("### Pre-annotated Dataset: Corn Leaf Disease")
    # Fulfills Requirement 5: Copy of the dataset link
    st.success("**Dataset Link:** https://universe.roboflow.com/final-enlye/corn-disease")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Requirement 1 Fulfillment:**
        - **Target Classes:** Corn Rust, Grey Leaf Spot, and Leaf Blight.
        - **Source:** Roboflow (Corn Disease.v4-final.yolov8.zip)
        - **Organization:** Data is properly partitioned into **Training**, **Validation**, and **Testing** folders.
        - **Configuration:** A `data.yaml` file was used to define paths and the three specific disease classes.
        """)
    
    with col2:
        if os.path.exists("val_batch0_labels.jpg"):
            st.image("val_batch0_labels.jpg", caption="Requirement 1: Sample of Annotated Training Batch (Leaf Diseases)")
        else:
            st.warning("val_batch0_labels.jpg not found in repository.")

# ==========================================
# SECTION 2: TRAINING RESULTS & METRICS
# ==========================================
elif choice == "2. Training Results & Metrics":
    st.title("📊 Training Performance (Requirement 2 & 5)")
    
    # Requirement 2: Recorded Performance Metrics (Extracted from your results.csv)
    # Final Epoch values: Precision: 0.9644, Recall: 0.9741, mAP50: 0.9784
    st.subheader("Final Recorded Metrics (Epoch 10)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", "0.9644", "+ High Accuracy")
    m2.metric("Recall", "0.9741", "+ High Coverage")
    m3.metric("mAP50", "0.9784", "Excellent Score")
    m4.metric("Train Box Loss", "0.3874", "Minimized")

    st.write("---")
    
    # Requirement 5: Colab Training Output Cell Screenshot
    st.subheader("📸 Colab Training Console Output")
    if os.path.exists("colab_training_output.png"):
        st.image("colab_training_output.png", caption="Requirement 5: Verification of training in Google Colab environment.", use_column_width=True)
    
    st.write("---")
    
    # Requirement 5: Results graphs and Confusion Matrix
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📈 Training Curves")
        if os.path.exists("results.png"):
            st.image("results.png", caption="Loss and Metric Progress")
    with col_b:
        st.subheader("🎯 Confusion Matrix")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Model Accuracy per Disease Class")

# ==========================================
# SECTION 3: IMAGE TESTING & OUTPUTS
# ==========================================
elif choice == "3. Image Testing & Outputs":
    st.title("🖼️ Image Testing (Requirement 3)")
    
    tab1, tab2 = st.tabs(["🚀 Run Live Detection", "📸 Sample Detection Output"])
    
    with tab1:
        st.write("Upload a corn leaf image to detect Rust, Grey Leaf Spot, or Blight.")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])
        if uploaded_file:
            model = load_yolo_model()
            image = Image.open(uploaded_file)
            
            # Predict
            results = model.predict(source=image, conf=0.25)
            
            # Requirement 3: Detect objects, display boxes, class labels, and confidence
            res_plotted = results[0].plot() 
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Live Detection Result", use_column_width=True)
    
    with tab2:
        st.write("Requirement 3: Presenting the detection output (Static Result).")
        if os.path.exists("detection_screenshot.png"):
            st.image("detection_screenshot.png", caption="Detection Output with Bounding Boxes, Class Labels, and Confidence Scores", use_column_width=True)

# ==========================================
# SECTION 4: COMPARATIVE ANALYSIS
# ==========================================
elif choice == "4. Comparative Analysis":
    st.title("🧠 Comparative Analysis (Requirement 4)")
    
    qa = {
        "1. What is object detection?": "Object detection is a computer vision task that identifies and locates objects within an image or video, determining both what the object is (classification) and where it is located (localization) via bounding boxes.",
        "2. How does YOLO perform object detection?": "YOLO (You Only Look Once) analyzes the entire image in a single pass, predicting bounding boxes and class probabilities simultaneously, which allows for real-time detection speeds.",
        "3. What is the role of a pre-annotated dataset?": "A pre-annotated dataset provides ground truth labels that the model uses to learn. It allows researchers to focus on training rather than manual data preparation.",
        "4. What do Precision, Recall, and mAP measure?": "Precision measures prediction accuracy. Recall measures the model's ability to find all actual objects. mAP (mean Average Precision) is the overall combined performance score.",
        "5. What challenges did you encounter during training?": "Challenges included distinguishing between Grey Leaf Spot and Leaf Blight due to their similar brownish lesion appearances, and managing path configurations in the Colab environment.",
        "6. How can object detection performance be improved?": "Performance can be improved by increasing the dataset size, using data augmentation techniques, and training for more epochs (e.g., 50+) for better convergence."
    }
    
    for q, a in qa.items():
        with st.expander(q):
            st.write(a)

# ==========================================
# SECTION 5: PROJECT SOURCE CODE
# ==========================================
elif choice == "5. Project Source Code":
    st.title("📄 Source Code (Requirement 5)")
    
    st.subheader("1. Training Script (Google Colab)")
    colab_train_code = """
from google.colab import drive
drive.mount('/content/drive')

!mkdir dataset
!unzip "/content/drive/MyDrive/Corn Disease.v4-final.yolov8.zip" -d /content/dataset

import warnings
warnings.filterwarnings('ignore')

!pip install ultralytics
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='/content/dataset/data.yaml',
    epochs=10,
    project='/content/drive/MyDrive/YOLO_Activity',
    name='results'
)
    """
    st.code(colab_train_code, language='python')
    
    st.subheader("2. Testing Script (Google Colab)")
    colab_test_code = """
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import cv2

model = YOLO('/content/drive/MyDrive/YOLO_Activity/results-7/weights/best.pt')

results = model.predict(source='/content/test_image.jpg', save=True, conf=0.25)

for r in results:
    res_plotted = r.plot()
    cv2_imshow(res_plotted)
    """
    st.code(colab_test_code, language='python')
    
    st.subheader("3. Web Application Code (app.py)")
    with open("app.py", "r") as f:
        st.code(f.read(), language='python')
