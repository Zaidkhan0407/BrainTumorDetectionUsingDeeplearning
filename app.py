import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('BrainTumor.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "MRI Image (No Brain Tumor)"
    elif classNo == 1:
        return "MRI Image (Yes Brain Tumor)"
    else:
        return "Not an MRI Image"

def detect_symmetry(image):
    if image.shape[0] < 10 or image.shape[1] < 10:
        return -1  # Non-MRI image

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (Canny)
    edges = cv2.Canny(gray_image, 100, 200)

    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate symmetry score
    symmetry_score = len(contours)  # Example: Use the number of contours as the symmetry score

    return symmetry_score

def getResult(img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    if result[0][0] > 0.5:
        return 0  # MRI
    elif result[0][1] > 0.5:
        return 1  # Brain Tumor
    else:
        return 2  # Non-MRI

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Get symmetry score
        symmetry_score = detect_symmetry(cv2.imread(file_path))

        # Get prediction result
        value = getResult(file_path)
        result = get_className(value) 

        # Return result and symmetry score as JSON
        data = {'result': result, 'symmetry_score': symmetry_score}
        print(data)
        return jsonify(data)

    return None


if __name__ == '__main__':
    app.run(debug=True)