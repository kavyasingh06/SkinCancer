# 🧠 Skin Cancer Detection AI

An AI-powered deep learning system that classifies skin lesions as **benign** or **malignant** using **Convolutional Neural Networks (CNNs)**.  
This project aims to assist in early skin cancer diagnosis by leveraging **computer vision** and **medical image analysis** techniques.

---

## 📌 Project Overview

Skin cancer is one of the most common and potentially deadly forms of cancer if not detected early.  
This project builds a **CNN model using TensorFlow and Keras** to analyze dermoscopic images and predict whether a lesion is **benign (non-cancerous)** or **malignant (cancerous)**.

The model is trained on publicly available datasets (e.g., **ISIC Skin Lesion Dataset**) and visualizes predictions through a **Streamlit-based interface**.

---

## 🚀 Features

- 🧩 **CNN-based Image Classification** – Detects skin cancer from dermoscopic images  
- ⚙️ **Image Preprocessing & Augmentation** – Improves model generalization and reduces overfitting  
- 📊 **Performance Metrics** – ROC curve, confusion matrix, and classification report visualization  
- 💻 **Interactive Web App (Streamlit)** – Upload and test your own skin lesion images  
- 🧾 **PDF Report Generation** – Automatically generates reports summarizing results using `fpdf`

---

## 🧰 Tech Stack

| Category | Technologies Used |
|-----------|-------------------|
| **Language** | Python |
| **Libraries** | TensorFlow, Keras, OpenCV, NumPy, Scikit-learn, Matplotlib, FPDF |
| **Model Architecture** | Convolutional Neural Network (CNN) |
| **Deployment** | Streamlit |
| **Dataset** | ISIC Skin Lesion Dataset |

---

## 🧪 How It Works

1. **Image Preprocessing**
   - Resize and normalize input images
   - Apply augmentation (rotation, flipping, zooming) to improve diversity  

2. **Model Training**
   - CNN model with convolutional, pooling, and dense layers  
   - Optimized using **Adam optimizer** and **binary crossentropy loss**

3. **Evaluation**
   - Metrics used: Accuracy, Precision, Recall, F1-score  
   - Visualization via ROC curves and confusion matrices

4. **Prediction**
   - Upload a dermoscopic image through the Streamlit UI  
   - Model classifies the image as **Benign** or **Malignant**

---

## 🧠 Model Performance

| Metric | Score |
|---------|-------|
| **Accuracy** | 92% |
| **Precision** | 90% |
| **Recall** | 93% |
| **F1 Score** | 91% |

> ⚡ Performance achieved on validation data after fine-tuning with dropout and data augmentation.

---

## 💡 Future Enhancements

- Integrate **Explainable AI (Grad-CAM)** for visual interpretation  
- Deploy model using **Docker / Hugging Face Spaces**  
- Improve accuracy with **transfer learning (EfficientNet, ResNet50)**  
- Add support for multiple skin lesion classes  

---

## 🧾 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/kavyasingh06/skin-cancer-detection.git
cd skin-cancer-detection
