import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained yoga pose detection model
model = load_model('yoga_pose_model.h5')

# Class names corresponding to model outputs
CLASS_NAMES = ["Downward Dog", "Goddess Pose" ,"Plank Pose", "Tree Pose" , "Warrior II"]

# Initialize Flask app
app = Flask(__name__)

# Configuration for upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Function to validate uploaded file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# File upload and pose prediction route
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file type or no file selected", 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Perform pose prediction
    predictions, pose_name = predict_pose(file_path)

    # Log predictions for debugging
    print(f"Raw Predictions: {predictions}")
    print(f"Predicted Pose: {pose_name}")

    return render_template('result.html', pose=pose_name, file_path=f'/{file_path}')


# Function to predict yoga pose from an image
def predict_pose(file_path):
    img = cv2.imread(file_path)

    # Debugging: Check the original image size
    print(f"Original Image Shape: {img.shape}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if necessary
    img = cv2.resize(img, (224, 224))  # Resize image to the required input shape
    img = img / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Debugging: Check the shape of image going into the model
    print(f"Preprocessed Image Shape: {img.shape}")

    # Model prediction
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions)  # Get the index of highest prediction
    predicted_pose = CLASS_NAMES[predicted_class_idx]

    return predictions, predicted_pose


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
