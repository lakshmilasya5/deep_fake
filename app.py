import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from pathlib import Path

# Load model
model = tf.keras.models.load_model("deepfake_model.h5")
IMG_SIZE = 128

def predict_image(image_path):
    img = Image.open(image_path)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    return "Fake" if pred <= 0.5 else "Real"

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fake_votes = 0
    total_frames = 0
    
    while total_frames < 20:  # Process max 20 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype("float32") / 255.0
        frame = np.expand_dims(frame, axis=0)
        
        pred = model.predict(frame, verbose=0)[0][0]
        if pred <= 0.5:
            fake_votes += 1
        total_frames += 1
    
    cap.release()
    return "Fake" if (fake_votes / total_frames) > 0.5 else "Real"

def process_file(file_path):
    if not file_path:
        return "No file uploaded"
    
    file_path = str(file_path)
    ext = Path(file_path).suffix.lower()
    
    if ext in ['.jpg', '.jpeg', '.png']:
        return predict_image(file_path)
    elif ext in ['.mp4', '.avi', '.mov']:
        return predict_video(file_path)
    else:
        return "Unsupported file type"

# Simple interface
interface = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload Image or Video", file_types=["image", "video"]),
    outputs="text",
    title="Deepfake Detection",
    description="Upload an image or video to check if it's real or fake"
)

interface.launch()
