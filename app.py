import cv2
import numpy as np
import onnxruntime as ort
import tkinter as tk
from PIL import Image, ImageTk

# Path to ONNX model
model_path = "best.onnx"

# Load model
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name

# Classes
class_names = [
    "batteries", "clothes", "e-waste", "glass",
    "light bulbs", "metal", "organic", "paper", "plastic"
]

# Webcam
cap = cv2.VideoCapture(0)

# Tkinter Window
window = tk.Tk()
window.title("Smart Waste Classifier")
window.geometry("800x600")

# Video Label
video_label = tk.Label(window)
video_label.pack()

# Prediction Label
pred_label = tk.Label(window, text="Waiting for prediction...",
                      font=("Helvetica", 20), fg="green")
pred_label.pack(pady=20)


def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Preprocess
    img = cv2.resize(frame, (input_shape[2], input_shape[3]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Inference
    outputs = session.run([output_name], {input_name: img})[0]
    pred_class = np.argmax(outputs, axis=1)[0]
    confidence = np.max(outputs)

    # Update UI
    label_text = f"Prediction: {class_names[pred_class]} ({confidence:.2f})"
    pred_label.config(text=label_text)

    # Convert frame for Tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Repeat
    window.after(10, update_frame)


update_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()

