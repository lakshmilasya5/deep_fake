import os
import logging
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Suppress TensorFlow logs (info, warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up basic logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Upload folder config
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB file limit

IMG_SIZE = 128
model = load_model('deepfake_model.h5')  # Load once

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def predict_image(filepath):
    try:
        img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)[0][0]
        return "Fake" if pred <= 0.5 else "Real"
    except Exception as e:
        logging.error(f"Error in image prediction: {e}")
        return "Error in image prediction"

def predict_video(filepath):
    try:
        cap = cv2.VideoCapture(filepath)
        frame_count = 0
        fake_votes = 0
        total_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= 10:  # Reduced to 10 frames
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame.astype("float32") / 255.0
            frame = np.expand_dims(frame, axis=0)
            pred = model.predict(frame)[0][0]
            if pred >= 0.5:
                fake_votes += 1
            total_frames += 1
            frame_count += 1

        cap.release()
        return "Fake" if fake_votes > total_frames / 2 else "Real"
    except Exception as e:
        logging.error(f"Error in video prediction: {e}")
        return "Error in video prediction"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                result = predict_video(filepath)
            elif filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                result = predict_image(filepath)
            else:
                result = "Unsupported file format"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run()
