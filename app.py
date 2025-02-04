from flask import Flask, render_template, request, redirect, url_for
import os
import cv2 as cv
import numpy as np
from sift_processor import SIFTProcessor
from werkzeug.utils import secure_filename
import uuid  # To generate unique filenames

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/images'  # Directory to store uploaded images

# Define allowed file extensions for security
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Create an instance of the SIFTProcessor class
sift_processor = SIFTProcessor()


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the home page with the upload form."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Handles the uploaded images, processes them using SIFT, and returns the result."""

    feature_type = request.form.get("feature_type")  # Get selected feature type
    files = request.files.getlist("image")  # Retrieve uploaded images

    # Ensure at least one file is uploaded
    if not files or not files[0].filename:
        return "No image file uploaded", 400

    file_paths = []  # List to store uploaded file paths

    # Process and save each uploaded file
    for file in files:
        if not allowed_file(file.filename):
            return "Invalid file format", 400  # Reject unsupported file formats

        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"  # Generate a unique filename
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)  # Save file to the server
        file_paths.append(upload_path)  # Store file path

    try:
        # Process the image based on the selected feature type
        if feature_type == "keypoints":
            result_path = sift_processor.process_image(file_paths[0], app.config['UPLOAD_FOLDER'], "keypoints")
        elif feature_type == "descriptors":
            result_path = sift_processor.process_image(file_paths[0], app.config['UPLOAD_FOLDER'], "descriptors")
        elif feature_type == "matching":
            # Ensure two images are uploaded for feature matching
            if len(file_paths) < 2:
                return "Please upload two images for matching", 400

            # Read images and process feature matching
            result_img = sift_processor.match_images(
                cv.imread(file_paths[0], cv.IMREAD_GRAYSCALE),
                cv.imread(file_paths[1], cv.IMREAD_GRAYSCALE)
            )
            # Save the matching result
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_matched.png")
            cv.imwrite(result_path, result_img)
        else:
            return "Invalid feature type", 400  # Handle incorrect selection
    except Exception as e:
        return str(e), 500  # Return error message if processing fails

    # Render the result page with the output image
    return render_template('result.html', result_path=result_path, feature_type=feature_type)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure the upload directory exists
    app.run(debug=True)  # Run Flask application in debug mode
