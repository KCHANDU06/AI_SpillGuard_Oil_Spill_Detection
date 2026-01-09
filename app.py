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
    thresh = st.sidebar.slider("Mask percentage threshold (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    k = st.sidebar.slider("Sigmoid steepness (higher = sharper)", min_value=0.1, max_value=2.0, value=0.6, step=0.1)

    # denoise mask with morphological operations to reduce false positives
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # filter small contours by area to avoid false positives
    min_contour_area_pct = st.sidebar.slider("Min contour area (% of image)", min_value=0.0, max_value=5.0, value=0.05, step=0.01)
    img_area = clean_mask.size
    filtered_mask = np.zeros_like(clean_mask)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= (min_contour_area_pct / 100.0) * img_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=-1)
    # if no contours passed the filter, fall back to cleaned mask
    if np.sum(filtered_mask == 255) == 0:
        filtered_mask = clean_mask.copy()

    # recompute percentage from filtered mask
    percentage = round((np.sum(filtered_mask == 255) / filtered_mask.size) * 100, 2)

    # map spill percentage to a probability using a sigmoid around the threshold
    prob_spill = float(1.0 / (1.0 + np.exp(-k * (percentage - thresh))))
    prob_no_spill = 1.0 - prob_spill

    # human-friendly prediction label (as requested)
    predicted_label = "Detect the oil spill" if prob_spill >= 0.5 else "No oil spill"

    # keep the existing status/risk/recommendation but derive color from prediction
    if prob_spill >= 0.5:
        status = "OIL SPILL DETECTED"
        risk = "HIGH"
        color = "red"
        recommendation = "Immediate containment and cleanup required."
    else:
        status = "NO SIGNIFICANT OIL SPILL"
        risk = "LOW"
        color = "green"
        recommendation = "Routine monitoring recommended."

    # ---------------- DISPLAY ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(original, use_container_width=True)

    with col2:
        st.subheader("Oil Spill Mask")
        st.image(mask, use_container_width=True)

    with col3:
        st.subheader("Overlay Result")
        overlay = original.copy()
        overlay[filtered_mask == 255] = [255, 0, 0]
        st.image(overlay, use_container_width=True)

    # ---------------- PREDICTION PROBABILITIES (VISUAL) ----------------
    st.markdown("## 📊 Prediction Confidence")
    pcol1, pcol2 = st.columns([2,1])
    with pcol1:
        st.write("**Predicted Label:**", predicted_label)
        st.write(f"**Oil Spill Probability:** {prob_spill*100:.2f}%")
        st.progress(int(prob_spill*100))
        st.write(f"**No Oil Spill Probability:** {prob_no_spill*100:.2f}%")
        st.progress(int(prob_no_spill*100))

    with pcol2:
        # small bar chart showing both probabilities
        fig, ax = plt.subplots(figsize=(3,2))
        labels = ["Oil Spill", "No Oil Spill"]
        vals = [prob_spill, prob_no_spill]
        ax.bar(labels, vals, color=["#d9534f", "#5cb85c"]) 
        ax.set_ylim(0,1)
        ax.set_ylabel('Probability')
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
        st.pyplot(fig)

    # ---------------- HEATMAP / CONTOUR VISUALIZATION ----------------
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.subheader("Mask (filtered)")
        st.image(filtered_mask, use_container_width=True)

    with viz_col2:
        st.subheader("Heatmap Overlay + Contours")
        # create heatmap from mask
        heat = cv2.normalize(filtered_mask, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat_overlay = cv2.addWeighted(original, 0.7, heat, 0.3, 0)
        # draw contours
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(heat_overlay, contours, -1, (255,255,255), 2)
        st.image(heat_overlay, use_container_width=True)

    # ---------------- EVALUATION (USER FEEDBACK) ----------------
    st.markdown("## 🧾 Evaluation / Ground Truth")
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {'TP':0, 'FP':0, 'FN':0, 'TN':0}

    gt = st.radio("Select ground-truth for this image:", ("Oil Spill", "No Oil Spill"))
    if st.button("Submit Ground Truth"):
        pred_is_spill = prob_spill >= 0.5
        gt_is_spill = True if gt == "Oil Spill" else False
        m = st.session_state.metrics
        if pred_is_spill and gt_is_spill:
            m['TP'] += 1
        elif pred_is_spill and not gt_is_spill:
            m['FP'] += 1
        elif not pred_is_spill and gt_is_spill:
            m['FN'] += 1
        else:
            m['TN'] += 1

        st.success("Ground truth submitted and metrics updated.")

    # show confusion numbers
    cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
    cm = st.session_state.metrics
    cm_col1.metric("TP", cm['TP'])
    cm_col2.metric("FP", cm['FP'])
    cm_col3.metric("FN", cm['FN'])
    cm_col4.metric("TN", cm['TN'])

    # (recall graph removed)

    # ---------------- REPORT ----------------
    st.markdown("## 📝 Detection Report")

    st.markdown(f"""
    <div class="report">
        <h3>Status: <span style="color:{color};">{status}</span></h3>
        <p><b>Risk Level:</b> {risk}</p>
        <p><b>Oil Spill Area:</b> {percentage}%</p>
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
Oil Spill Coverage: {percentage}%

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
