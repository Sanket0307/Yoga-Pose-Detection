# Yoga-Pose-Detection
## Project Overview
The Yoga Pose Detection project classifies images of yoga poses into predefined categories using a deep learning-based approach. It utilizes computer vision techniques and a Convolutional Neural Network (CNN) to identify and categorize poses. An interactive web interface allows users to upload images and receive predictions in real-time.

***Approach***
The project follows a structured approach from data collection to deployment.

_1. Data Collection_
Images of various yoga poses were gathered from publicly available datasets and curated manually.
Classes included:
  -Warrior II
  -Tree Pose
  -Downward Dog
  -Goddess Pose
  -Plank Pose

_2. Data Preprocessing_
Image Resizing: All images were resized to 224x224 pixels to match the input requirements of the CNN model.
Normalization: Pixel values were scaled to the range [0, 1] to improve model performance.
Data Augmentation: Applied techniques to increase data diversity and reduce overfitting:
  -Random rotations
  -Flipping (horizontal and vertical)
  -Zoom and brightness adjustments

_3. Model Design_
A Convolutional Neural Network (CNN) architecture was selected for its robustness in image classification tasks.
Architecture Details
  -Input Layer: Accepts images of shape 224x224x3.
  -Convolutional Layers: Extract features using multiple filters.
  -Batch Normalization: Normalizes intermediate inputs to stabilize learning.
  -Pooling Layers: Reduces spatial dimensions for computational efficiency.
  -Fully Connected Layers: Dense layers process extracted features for classification.
  -Output Layer:
    -Contains five neurons (one for each pose).
    -Uses the softmax activation function for multi-class classification.
_Model Parameters_
  -Loss Function: Categorical Crossentropy
  -Optimizer: Adam optimizer for efficient learning

_4. Web Application_
  -Backend: Built using Flask to handle image uploads and predictions.
  -Frontend: Features an interactive and visually appealing interface for:
    -Uploading images
    -Displaying predictions dynamically
  -Integration: The trained model is seamlessly integrated with the Flask app for real-time inference.
**Results**
_Accuracy_:
  -Training Accuracy: ~96%
  -Validation Accuracy: ~92%
  -Test Accuracy: ~90% on unseen data.
***Acknowledgments***
  -Inspiration from fitness applications and computer vision advancements.
  -Special thanks to publicly available yoga pose datasets used for training.
**Next Steps**
_Model Improvements:
Experiment with tra_nsfer learning using pre-trained models like ResNet or EfficientNet.
Increase the dataset size for better generalization.
_Real-Time Detection:_
Integrate with OpenCV or Mediapipe for real-time yoga pose detection via webcams.
_Enhance Web UI:_
Add interactive feedback for incorrect poses.
Include tips for improving posture based on pose predictions.
_Mobile Deployment:_
Convert the model to TensorFlow Lite for use on mobile devices.
