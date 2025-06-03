# 🍎🍌 Fresh vs Rotten Fruit Classifier (VGG16 & ResNet)

This project uses **transfer learning** to classify fruit images into **fresh** or **rotten** categories with high accuracy.

## 🧠 Project Overview

The goal is to classify fruit quality using pre-trained models:
- **VGG16**: 95% accuracy
- **ResNet50**: 99% accuracy (slight overfitting observed)

## 🛠️ Technologies Used
- Python
- Keras (with TensorFlow backend)
- OpenCV
- VGG16, ResNet50 (from keras.applications)

## 📂 Model Summary

- Input Shape: (224, 224, 3)
- Preprocessing: Image resizing, normalization, augmentation
- Layers: Transfer learning base (frozen) + new classifier head
- Loss: Binary Crossentropy
- Optimizer: Adam

## 📊 Results

| Model     | Accuracy | Comments                 |
|-----------|----------|--------------------------|
| VGG16     | 95%      | Good generalization      |
| ResNet50  | 99%      | Excellent, but may overfit|

## 🎯 Use Case

- Real-time fruit quality check in supermarkets, farms, and supply chains.

## 📚 Key Learnings

- Transfer learning is powerful for image classification with limited data
- Deeper networks like ResNet generalize better — but need careful tuning
- Importance of early stopping & validation monitoring

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/fruit-classifier
cd fruit-classifier
pip install -r requirements.txt
python train.py
