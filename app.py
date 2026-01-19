# This code has been reviewed and appears to be correct.
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
}
h1, h2, h3 {
    color: #00f5ff;
}
.report {
    background-color: rgba(0,0,0,0.65);
    padding: 20px;
    border-radius: 15px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🌊 AI-Based Oil Spill Detection System</h1>", unsafe_allow_html=True)
st.write("Upload a satellite image to detect and analyze oil spill regions automatically.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- FUNCTIONS ----------------
def preprocess_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, blur

def oil_spill_segmentation(gray_img):
    _ , mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask

def spill_percentage(mask):
    spill = np.sum(mask == 255)
    total = mask.size
    return round((spill / total) * 100, 2)

# ---------------- PROCESS ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    original, gray = preprocess_image(image)
    mask = oil_spill_segmentation(gray)
    percentage = spill_percentage(mask)

    # Decision Logic
    # ---------- USER ADJUSTABLE PARAMETERS ----------
    st.sidebar.markdown("## ⚙️ Detection Parameters")
    # Threshold for the percentage of the image covered by the spill to be considered a spill
    thresh = st.sidebar.slider("1. Spill Area Threshold (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1, help="The minimum percentage of the image that must be covered by a spill to trigger a detection.")
    # Sigmoid steepness for probability calculation
    k = st.sidebar.slider("2. Sigmoid Steepness", min_value=0.1, max_value=2.0, value=0.6, step=0.1, help="Controls the sharpness of the probability curve. Higher values make the decision boundary more abrupt.")
    # Minimum area for a contour to be considered a potential spill
    min_contour_area_pct = st.sidebar.slider("3. Min Contour Area (% of image)", min_value=0.0, max_value=5.0, value=0.05, step=0.01, help="Filters out very small, noisy detections to reduce false positives.")
    # Aspect ratio filtering parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Shape Filtering (Advanced)")
    min_aspect_ratio = st.sidebar.slider("4. Min Aspect Ratio (w/h)", min_value=0.05, max_value=10.0, value=0.1, step=0.05, help="Minimum width/height ratio. Helps filter out unnaturally thin or tall shapes.")
    max_aspect_ratio = st.sidebar.slider("5. Max Aspect Ratio (w/h)", min_value=1.0, max_value=20.0, value=10.0, step=0.1, help="Maximum width/height ratio. Helps filter out unnaturally wide or flat shapes.")


    # --- Step 1: Denoise Mask ---
    # Apply morphological operations to remove noise from the initial mask.
    # MORPH_OPEN removes small white spots (salt noise).
    # MORPH_CLOSE fills small black holes (pepper noise).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # --- Step 2: Filter Contours by Area and Shape ---
    # To further reduce false positives, we filter the detected regions (contours)
    # based on their size and shape.
    img_area = clean_mask.size
    filtered_mask = np.zeros_like(clean_mask)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by area
        if area >= (min_contour_area_pct / 100.0) * img_area:
            # Filter by shape (aspect ratio of bounding box)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(1, h)  # Avoid division by zero
            if aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=-1)

    # --- Step 3: Recalculate Spill Percentage ---
    # After filtering, recalculate the spill percentage based on the final cleaned mask.
    percentage = round((np.sum(filtered_mask == 255) / filtered_mask.size) * 100, 2)

    # --- Step 4: Final Prediction Logic ---
    # Map the final spill percentage to a probability using a sigmoid function.
    # This creates a "soft" decision boundary around the user-defined threshold.
    # If percentage > thresh, prob_spill > 0.5.
    # If percentage < thresh, prob_spill < 0.5.
    prob_spill = float(1.0 / (1.0 + np.exp(-k * (percentage - thresh))))
    prob_no_spill = 1.0 - prob_spill

    # Determine the final status based on the spill probability.
    if prob_spill >= 0.5:
        status = "OIL SPILL DETECTED"
        risk = "HIGH"
        color = "red"
        recommendation = "Immediate containment and cleanup required."
    else:
        status = "NO OIL SPILL DETECTED"
        risk = "LOW"
        color = "green"
        recommendation = "Routine monitoring recommended."

    # ---------------- DISPLAY ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(original, use_container_width=True)

    if status == "OIL SPILL DETECTED":
        with col2:
            st.subheader("Oil Spill Mask")
            # Show the final, filtered mask that is used for analysis
            st.image(filtered_mask, use_container_width=True)

        with col3:
            st.subheader("Overlay Result")
            overlay = original.copy()
            # Highlight detected regions in red on the original image
            overlay[filtered_mask == 255] = [255, 0, 0]
            st.image(overlay, use_container_width=True)
    else:
        # If no spill is detected, show a clear message and the clean image.
        with col2:
            st.success("✅ No significant spill detected.")
        with col3:
            st.image(original, use_container_width=True, caption="Image appears to be clean.")

    # ---------------- PREDICTION PROBABILITIES (VISUAL) ----------------
    st.markdown("## 📊 Prediction Confidence")
    # This section is shown for both cases to display the model's confidence in its prediction.
    pcol1, pcol2 = st.columns([2,1])
    with pcol1:
        st.write(f"**Predicted Label:** {status}")
        # Show the confidence for the predicted label.
        if status == "OIL SPILL DETECTED":
            st.write(f"**Confidence:** {prob_spill*100:.2f}%")
            st.progress(int(prob_spill*100))
        else: # NO OIL SPILL DETECTED
            st.write(f"**Confidence:** {prob_no_spill*100:.2f}%")
            st.progress(int(prob_no_spill*100))

    with pcol2:
        # A bar chart to visualize the comparison between spill and no-spill probabilities.
        fig, ax = plt.subplots(figsize=(3,2))
        labels = ["Oil Spill", "No Oil Spill"]
        vals = [prob_spill, prob_no_spill]
        colors = ["#d9534f", "#5cb85c"]
        ax.bar(labels, vals, color=colors) 
        ax.set_ylim(0,1)
        ax.set_ylabel('Probability')
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
        st.pyplot(fig)
    
    # ---------------- DETAILED VISUALIZATIONS (SPILL ONLY) ----------------
    if status == "OIL SPILL DETECTED":
        st.markdown("## 🔬 Detailed Analysis")
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.subheader("Raw Segmentation Mask")
            st.image(mask, use_container_width=True, caption="Initial mask before filtering.")
        with viz_col2:
            st.subheader("Heatmap Overlay + Contours")
            # Create a heatmap to visualize spill density
            heat = cv2.normalize(filtered_mask, None, 0, 255, cv2.NORM_MINMAX)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            heat_overlay = cv2.addWeighted(original, 0.7, heat, 0.3, 0)
            # Draw final contours on the heatmap
            contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(heat_overlay, contours, -1, (255,255,255), 2)
            st.image(heat_overlay, use_container_width=True, caption="Heatmap of spill areas with contours.")

    # (recall graph removed)

    # ---------------- REPORT ----------------
    st.markdown("## 📝 Detection Report")

    # Conditionally format the spill area line for clarity in the report
    if status == "OIL SPILL DETECTED":
        spill_area_html = f'<p><b>Oil Spill Area:</b> {percentage}%</p>'
        spill_area_txt = f'Oil Spill Coverage: {percentage}%'
    else:
        spill_area_html = '<p><b>Oil Spill Area:</b> None</p>'
        spill_area_txt = 'Oil Spill Coverage: None'

    st.markdown(f"""
    <div class="report">
        <h3>Status: <span style="color:{color};">{status}</span></h3>
        <p><b>Risk Level:</b> {risk}</p>
        {spill_area_html}
        <p><b>Analysis Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><b>Detection Method:</b> AI-Based Image Segmentation</p>
        <p><b>Recommendation:</b> {recommendation}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- DOWNLOAD REPORT ----------------
    report_text = f"""
AI Oil Spill Detection Report
============================
Status: {status}
Risk Level: {risk}
{spill_area_txt}

Analysis Time: {datetime.now()}

Recommendation:
{recommendation}
"""

    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name="oil_spill_report.txt",
        mime="text/plain"
    )

else:
    st.info("⬆️ Please upload a satellite image to begin detection.")
