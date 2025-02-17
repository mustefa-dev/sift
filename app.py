from flask import Flask, render_template, request, redirect, url_for
import os
import cv2 as cv
import numpy as np
from sift_processor import SIFTProcessor
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
sift_processor = SIFTProcessor()

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    feature_type = request.form.get("feature_type")
    files = request.files.getlist("image")

    if not files or not files[0].filename:
        return "No image file uploaded", 400

    file_paths = []
    for file in files:
        if not allowed_file(file.filename):
            return "Invalid file format", 400

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        file_paths.append(upload_path)

    try:
        if feature_type == "keypoints":
            result_path = sift_processor.process_image(file_paths[0], app.config['UPLOAD_FOLDER'], "keypoints")
        elif feature_type == "descriptors":
            result_path = sift_processor.process_image(file_paths[0], app.config['UPLOAD_FOLDER'], "descriptors")
        elif feature_type == "matching":
            if len(file_paths) < 2:
                return "Please upload two images for matching", 400
            result_img = sift_processor.match_images(cv.imread(file_paths[0], cv.IMREAD_GRAYSCALE),
                                                     cv.imread(file_paths[1], cv.IMREAD_GRAYSCALE))
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_matched.png")
            cv.imwrite(result_path, result_img)
        else:
            return "Invalid feature type", 400
    except Exception as e:
        return str(e), 500

    return render_template('result.html', result_path=result_path, feature_type=feature_type)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
