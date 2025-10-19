import streamlit as st
import cv2
import numpy as np
from brain_tumor_detector import detect_and_segment
from PIL import Image
from io import BytesIO
import os

@st.cache_data
def process_image(img_path):
    return detect_and_segment(img_path)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection & Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .detection-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🧠 Brain Tumor Detection & Segmentation")
st.subheader("*AI-Powered Medical Image Analysis using YOLO11 + SAM2*")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.header("About")
    st.markdown("""
    This application uses advanced deep learning models:
    - **YOLO11n**: For tumor detection
    - **SAM2**: For precise segmentation
    
    **Detectable Tumor Types:**
    - Glioma
    - Meningioma
    - Pituitary
    - No Tumor
    """)
    
    st.divider()
    
    st.header("Model Info")
    st.metric("Model Accuracy", "81.5%", "mAP50")
    st.metric("Inference Speed", "~11ms", "per image")
    
    st.divider()
    st.markdown("**⚠️ Disclaimer:** This tool is for research purposes only and should not replace professional medical diagnosis.")

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload MRI Image")
    
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI image for analysis"
    )
    
    if uploaded_file:
        # Display original image
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original MRI Image", use_container_width=True)
        
        # Save uploaded file
        img_path = os.path.join("uploads", uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        st.info("Please upload an MRI image to begin analysis")

with col2:
    if uploaded_file:
        st.subheader("🔍 Analysis Results")
        
        # Run detection
        with st.spinner("🔄 Processing image... This may take a few seconds"):
            processed_img, results = process_image(img_path)
        
        # Display processed image
        st.image(
            cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
            caption="Detected & Segmented Image",
            use_container_width=True
        )
        
        # Save output
        output_path = os.path.join("outputs", f"output_{uploaded_file.name}")
        cv2.imwrite(output_path, processed_img)
        
        # Prepare download
        pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="⬇️ Download Processed Image",
            data=byte_im,
            file_name=f"segmented_{uploaded_file.name}",
            mime="image/png",
            use_container_width=True
        )

# Detection Results Section
if uploaded_file and results:
    st.divider()
    st.subheader("📊 Detection Summary")
    
    if results:
        # Create metrics row
        metric_cols = st.columns(len(results))
        
        for idx, det in enumerate(results):
            with metric_cols[idx]:
                confidence_color = "🟢" if det['confidence'] > 0.7 else "🟡" if det['confidence'] > 0.5 else "🔴"
                st.markdown(f"""
                    <div class="stats-box">
                        <h3 style="margin: 0;">{confidence_color} {det['class'].replace('_', ' ').title()}</h3>
                        <p style="font-size: 2rem; margin: 0.5rem 0; font-weight: bold;">{det['confidence']*100:.1f}%</p>
                        <p style="margin: 0; font-size: 0.9rem;">Confidence</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Detailed results
        st.subheader("📋 Detailed Detections")
        for idx, det in enumerate(results, 1):
            with st.expander(f"Detection {idx}: {det['class'].replace('_', ' ').title()}", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Tumor Type", det['class'].replace('_', ' ').title())
                    st.metric("Confidence Score", f"{det['confidence']*100:.2f}%")
                with col_b:
                    bbox = det['bbox']
                    st.metric("Bounding Box", f"[{int(bbox[0])}, {int(bbox[1])}]")
                    width = int(bbox[2] - bbox[0])
                    height = int(bbox[3] - bbox[1])
                    st.metric("Detection Size", f"{width}×{height}px")
    else:
        st.info("✅ No tumors detected in this image")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 1rem;'>
        <p>Powered by YOLO11n & SAM2 | Made with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)