# 🧘‍♂️ Yoga Pose Detection

An AI-powered web app that classifies images of yoga poses using computer vision and deep learning. The model is built using a Convolutional Neural Network (CNN) and integrated into a Flask-based web application for real-time predictions.

---

## 🌟 Project Overview

This project aims to detect and classify popular yoga poses from user-uploaded images. Using a CNN model and an interactive web interface, users can receive instant feedback on their yoga pose classification.

---

## 🚀 Approach

### 1. 📸 Data Collection

Images of yoga poses were collected from publicly available datasets and manually curated. The following classes were included:

- Warrior II  
- Tree Pose  
- Downward Dog  
- Goddess Pose  
- Plank Pose  

### 2. 🧹 Data Preprocessing

- **Image Resizing:** All images resized to `224x224` pixels.  
- **Normalization:** Pixel values scaled to [0, 1].  
- **Data Augmentation:**  
  - Random rotations  
  - Horizontal and vertical flipping  
  - Zoom and brightness adjustments  

### 3. 🧠 Model Design

A CNN model was used for robust image classification.

- **Input Layer:** 224x224x3 image shape  
- **Convolutional Layers:** Feature extraction  
- **Batch Normalization:** Stabilizes learning  
- **Pooling Layers:** Reduces dimensionality  
- **Dense Layers:** Fully connected layers for classification  
- **Output Layer:**  
  - 5 neurons (one per pose)  
  - Softmax activation  

**Model Config:**  
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  

### 4. 🌐 Web Application

- **Backend:** Flask (handles image upload and inference)  
- **Frontend:** Interactive UI to:
  - Upload yoga pose images  
  - Display predictions in real time  

- **Integration:** The trained model is integrated for seamless live inference.

---

## 📊 Results

- **Training Accuracy:** ~96%  
- **Validation Accuracy:** ~92%  
- **Test Accuracy:** ~90% on unseen data  

---

## 🙏 Acknowledgments

- Inspired by fitness tech and computer vision use cases  
- Dataset used: [Yoga Poses Dataset (Kaggle)](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset/data?select=DATASET)

---

## 🔮 Next Steps

### 🔧 Model Improvements
- Experiment with **transfer learning** using models like ResNet or EfficientNet  
- Increase dataset size for better generalization  

### 📸 Real-Time Detection
- Integrate **OpenCV** or **MediaPipe** for webcam-based real-time detection  

### 💻 UI Enhancements
- Add feedback on incorrect poses  
- Provide tips for improving posture  

### 📱 Mobile Deployment
- Convert model to **TensorFlow Lite** for mobile app integration  

---

## 📁 Project Structure (optional)

