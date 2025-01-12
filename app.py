from flask import Flask, render_template, request, redirect, url_for
import os
from sift_processor import SIFTProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/images'
sift_processor = SIFTProcessor()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    feature_type = request.form.get("feature_type")
    files = request.files.getlist("image")

    if len(files) == 0 or not files[0].filename:
        return "No image file uploaded", 400

    # Save uploaded files
    file_paths = []
    for file in files:
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(upload_path)
        file_paths.append(upload_path)

    # Process images
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
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], "matched_output.png")
            cv.imwrite(result_path, result_img)
        else:
            return "Invalid feature type", 400
    except Exception as e:
        return str(e), 500

    return render_template('result.html', result_path=result_path, feature_type=feature_type)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
