import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI 8.0: Corn Object Detection Project",
    page_icon="🌽",
    layout="wide"
)

# --- LOAD TRAINED MODEL ---
@st.cache_resource
def load_yolo_model():
    # Uses the 'best.pt' weights generated from your Colab training
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
    st.write("### Pre-annotated Dataset: Corn Detection")
    st.info("**Dataset Source:** [Paste your Roboflow/Kaggle Link here]")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Requirement 1 Fulfillment:**
        - **Source:** Roboflow (Corn.v6i.yolov8.zip)
        - **Format:** YOLOv8 (Standard images and txt annotations).
        - **Subsets:** Properly organized into **Training**, **Validation**, and **Testing** sets.
        - **Configuration:** A `data.yaml` file was used to define the paths and class labels for the corn features.
        """)
    
    with col2:
        if os.path.exists("val_batch0_labels.jpg"):
            st.image("val_batch0_labels.jpg", caption="Requirement 1: Sample of Pre-annotated Dataset (Batch Labels)")
        else:
            st.warning("val_batch0_labels.jpg not found in repository.")

# ==========================================
# SECTION 2: TRAINING RESULTS & METRICS
# ==========================================
elif choice == "2. Training Results & Metrics":
    st.title("📊 Training Performance (Requirement 2 & 5)")
    st.write("Below are the recorded performance metrics and screenshots from the training process.")

    # Recorded Metrics from results.csv
    if os.path.exists("results.csv"):
        df = pd.read_csv("results.csv")
        df.columns = df.columns.str.strip() 
        final = df.iloc[-1]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Precision", f"{final['metrics/precision(B)']:.4f}")
        m2.metric("Final Recall", f"{final['metrics/recall(B)']:.4f}")
        m3.metric("Final mAP50", f"{final['metrics/mAP50(B)']:.4f}")
        m4.metric("Last Box Loss", f"{final['train/box_loss']:.4f}")
    
    st.write("---")
    
    # Requirement 5: Colab Training Output Screenshot
    st.subheader("📸 Colab Training Console Output")
    st.write("This screenshot verifies the model training was performed using a T4 GPU on Google Colab.")
    if os.path.exists("colab_training_output.png"):
        st.image("colab_training_output.png", caption="Requirement 2 & 5: Screenshot of Colab Training Cell Output", use_column_width=True)
    
    st.write("---")
    
    # Requirement 5: Results graphs
    st.subheader("📈 Training Performance Graphs")
    if os.path.exists("results.png"):
        st.image("results.png", caption="Requirement 5: YOLOv8 Training Results (Loss and Metrics over 10 Epochs)")

# ==========================================
# SECTION 3: IMAGE TESTING & OUTPUTS
# ==========================================
elif choice == "3. Image Testing & Outputs":
    st.title("🖼️ Image Testing (Requirement 3)")
    
    tab1, tab2 = st.tabs(["🚀 Run Live Detection", "📸 Saved Detection Output"])
    
    with tab1:
        st.write("Upload a corn image to test the model on unseen data.")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])
        if uploaded_file:
            model = load_yolo_model()
            image = Image.open(uploaded_file)
            
            # Predict
            results = model.predict(source=image, conf=0.25)
            
            # Show live result
            res_plotted = results[0].plot() 
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Live Detection Result", use_column_width=True)
            
            st.success("Successfully detected objects in the uploaded image.")
    
    with tab2:
        st.write("Requirement 3: Presentation of the detection output (Static Result).")
        if os.path.exists("detection_screenshot.png"):
            st.image("detection_screenshot.png", caption="Requirement 3: Detection with Bounding Boxes, Predicted Class Labels, and Confidence Scores", use_column_width=True)

# ==========================================
# SECTION 4: COMPARATIVE ANALYSIS
# ==========================================
elif choice == "4. Comparative Analysis":
    st.title("🧠 Comparative Analysis (Requirement 4)")
    st.write("Answering the evaluation questions based on the activity module.")
    
    qa = {
        "1. What is object detection?": "Object detection is a computer vision task that enables a computer to identify and locate objects within an image or video, determining both what the object is and where it is located via bounding boxes.",
        "2. How does YOLO perform object detection?": "YOLO (You Only Look Once) analyzes the entire image in a single pass, predicting bounding boxes and class probabilities simultaneously, which allows for real-time detection speeds.",
        "3. What is the role of a pre-annotated dataset?": "A pre-annotated dataset provides ground truth labels (bounding boxes and classes) that the model uses to learn. It allows learners to focus on training rather than manual data preparation.",
        "4. What do Precision, Recall, and mAP measure?": "Precision measures the accuracy of the predictions. Recall measures the model's ability to find all actual objects. mAP (mean Average Precision) is a combined score of overall performance.",
        "5. What challenges did you encounter during training?": "Challenges included configuring the dataset paths in Google Colab, managing GPU memory, and fine-tuning confidence thresholds for better detection results on corn features.",
        "6. How can object detection performance be improved?": "Performance can be improved by increasing the dataset size, using data augmentation techniques (flips, rotations), and training for more epochs (e.g., 50 or 100)."
    }
    
    for q, a in qa.items():
        with st.expander(q):
            st.write(a)

# ==========================================
# SECTION 5: PROJECT SOURCE CODE
# ==========================================
elif choice == "5. Project Source Code":
    st.title("📄 Source Code (Requirement 5)")
    
    # Training Code Block
    st.subheader("1. Training Script (Google Colab)")
    st.write("Code used to build and train the model using a T4 GPU:")
    colab_train_code = """
from google.colab import drive
drive.mount('/content/drive')

!mkdir dataset
!unzip /content/drive/MyDrive/Corn.v6i.yolov8.zip -d /content/dataset

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
    
    # Testing Code Block
    st.subheader("2. Testing Script (Google Colab)")
    st.write("Code used for initial image testing and verification in Colab:")
    colab_test_code = """
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import cv2

# Load the best weights from the training run
model = YOLO('/content/drive/MyDrive/YOLO_Activity/results-4/weights/best.pt')

# Perform prediction on a test image
results = model.predict(source='/content/drive/MyDrive/YOLO_Activity/my_test_image.jpg', save=True, conf=0.25)

# Display the result
for r in results:
    res_plotted = r.plot()
    cv2_imshow(res_plotted)

print(f"Result saved to: {results[0].save_dir}")
    """
    st.code(colab_test_code, language='python')
    
    # Web App Code Block
    st.subheader("3. Web Application Code (app.py)")
    st.write("This is the code for this Streamlit dashboard:")
    with open("app.py", "r") as f:
        st.code(f.read(), language='python')
