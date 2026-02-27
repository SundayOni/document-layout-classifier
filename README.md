# Document Layout Classifier

A convolutional neural network (CNN) that classifies scanned document images into three categories — **email**, **resume**, and **scientific publication** — using transfer learning with ResNet18.

---

## 📌 Overview

This project demonstrates end-to-end computer vision for document understanding:
- Exploratory data analysis of document image properties
- Image preprocessing and data augmentation
- Transfer learning with a pre-trained ResNet18 model
- Evaluation with classification report, confusion matrix and prediction visualisation

---

## 📂 Dataset

- **Source**: [Document Classification Dataset](https://www.kaggle.com/datasets/ritvik1909/document-classification-dataset) (Kaggle)
- **Size**: 165 scanned document images (55 per class)
- **Classes**: email, resume, scientific_publication
- **Format**: PNG images, ~754×1000 pixels

> Raw images are not included in this repository. Download the dataset from Kaggle and place it in the `data/` folder.

---

## 🏗️ Project Structure
```
document-layout-classifier/
├── data/                    # Document images (not tracked in git)
│   ├── email/
│   ├── resume/
│   └── scientific_publication/
├── models/                  # Saved model weights (not tracked in git)
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   └── 02_modelling.ipynb
├── src/
│   └── predict.py           # Run inference on new images
├── requirements.txt
└── .gitignore
```

---

## 🤖 Model

- **Architecture**: ResNet18 pre-trained on ImageNet
- **Approach**: Transfer learning — all layers frozen except the final fully connected layer
- **Input size**: 224×224 pixels
- **Training**: 15 epochs, Adam optimiser, CrossEntropyLoss, learning rate 0.001
- **Augmentation**: Random horizontal flip, rotation (±5°), colour jitter

---

## 📈 Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Email | 0.89 | 1.00 | 0.94 |
| Resume | 0.86 | 0.67 | 0.75 |
| Scientific Publication | 0.78 | 0.88 | 0.82 |
| **Macro Average** | **0.84** | **0.85** | **0.84** |

**Test Accuracy: 84%**

---

## 🚀 Getting Started
```bash
# Clone the repo
git clone https://github.com/SundayOni/document-layout-classifier.git
cd document-layout-classifier

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
kaggle datasets download -d ritvik1909/document-classification-dataset --path data/ --unzip

# Launch Jupyter
jupyter notebook
```

---

## 🔍 Key Findings

- Transfer learning is highly effective even with small datasets — 84% accuracy from just 165 images
- Email is the most visually distinct class (perfect recall), consistent with EDA showing emails have the highest brightness and most whitespace
- Resume and scientific publication are occasionally confused due to similar dense, structured layouts

---

## 🚀 Potential Improvements

- Collect more training data (500+ images per class)
- Fine-tune deeper ResNet layers rather than just the final layer
- Try larger architectures such as ResNet50 or EfficientNet
- Deploy as a Streamlit web app for live document classification