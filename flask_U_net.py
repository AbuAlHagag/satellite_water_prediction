from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import tifffile as tiff
from PIL import Image

# Keras
from keras.models import load_model

# Flask utils
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Define a Flask app
app = Flask(__name__)

# Load your model (ensure correct path)
MODEL_PATH = 'U_net_model_1.h5'
model = load_model(MODEL_PATH)

# Directory for saving images
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the upload and result directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

def model_predict(img_path, model):
    # Read and preprocess the image
    image = tiff.imread(img_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    preds = model.predict(image)
    return preds[0]  # Remove the batch dimension

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['imagefile']

        # Save the file to ./uploads
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Handle prediction array
        # Assuming preds is a 3D array (batch_size, height, width) and you need to handle it
        if len(preds.shape) == 3:
            preds = np.squeeze(preds)  # Remove any single-dimensional entries from the shape
        if preds.ndim == 2:  # Ensure it's 2D (height, width)
            # Convert prediction (grayscale) to an image and save it
            result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result.png')
            result_image = Image.fromarray((preds * 255).astype(np.uint8))  # Convert float image to uint8
            result_image.save(result_image_path)
            return render_template('index.html', result_image='results/result.png')
        else:
            return 'Prediction output shape is not valid.'

    return None

@app.route('/results/<filename>')
def send_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
