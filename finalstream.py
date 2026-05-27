import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort

# ----------------------
# Load ONNX model
# ----------------------
MODEL_PATH = "best.onnx"  # adjust path
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ----------------------
# Classes (fixed order) & Dustbin mapping
# ----------------------
CLASSES = ["batteries", "clothes", "e-waste", "glass", "light bulbs",
           "metal", "organic", "paper", "plastic"]

DUSTBIN_MAP = {
    "batteries": "⚡ Hazardous Waste Bin",
    "clothes": "👕 Textile Bin",
    "e-waste": "💻 E-Waste Bin",
    "glass": "🍾 Glass Bin",
    "light bulbs": "💡 Hazardous Waste Bin",
    "metal": "🔩 Recyclable Bin",
    "organic": "🌱 Organic/Compost Bin",
    "paper": "📄 Paper Bin",
    "plastic": "🍼 Plastic Bin"
}

# ----------------------
# Streamlit Setup
# ----------------------
st.set_page_config(page_title="♻️ Smart Waste Detector", layout="wide")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Webcam not accessible.")
        break

    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (224, 224))
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Run inference
    preds = session.run([output_name], {input_name: input_tensor})[0][0]
    probs = np.exp(preds) / np.sum(np.exp(preds))  # softmax
    top_class = np.argmax(probs)
    confidence = probs[top_class]

    # Labels
    label = CLASSES[top_class]
    bin_type = DUSTBIN_MAP[label]

    # -------------------
    # Overlay UI on frame
    # -------------------
    h, w, _ = frame.shape

    # Top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Title
    cv2.putText(frame, "♻️ Smart Waste Detector", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

    # Detected label
    cv2.putText(frame, f"Detected: {label}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Confidence bar
    bar_x, bar_y = 20, 110
    bar_w, bar_h = int(confidence * (w - 40)), 20
    cv2.rectangle(frame, (bar_x, bar_y), (w - 20, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), -1)
    cv2.putText(frame, f"{confidence*100:.1f}%", (bar_x + 10, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Dustbin suggestion
    cv2.putText(frame, f"Suggested Bin: {bin_type}", (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 215, 0), 2)

    # Stream
    FRAME_WINDOW.image(frame, channels="BGR")

