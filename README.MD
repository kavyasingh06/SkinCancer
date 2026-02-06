🧠 Skin Cancer Detection AI

An AI-powered deep learning system that classifies skin lesions as benign or malignant using Convolutional Neural Networks (CNNs).
This project demonstrates the application of computer vision and deep learning techniques to assist in early-stage skin cancer screening.

📌 Project Overview

Skin cancer is one of the most common and potentially life-threatening cancers if not detected early.
This project leverages TensorFlow and Keras to build a CNN-based classification pipeline that analyzes dermoscopic images and predicts whether a skin lesion is benign (non-cancerous) or malignant (cancerous).

The system covers the complete ML lifecycle — data preprocessing, model training, fine-tuning, evaluation, and inference — with a focus on reproducibility and deployment readiness.

🚀 Features

🧩 CNN-based Image Classification for skin cancer detection

⚙️ Image Preprocessing & Augmentation to reduce overfitting

📊 Model Evaluation Metrics including accuracy, precision, recall, F1-score

🧾 Batch Inference & PDF Report Generation using fpdf

🔁 Fine-tuned Transfer Learning using VGG16

🧰 Tech Stack
Category	Technologies Used
Language	Python
Libraries	TensorFlow, Keras, OpenCV, NumPy, Scikit-learn, Matplotlib, FPDF
Model Architecture	VGG16 (Transfer Learning)
Dataset	ISIC / HAM10000 Skin Lesion Dataset
Interface	Batch-based Inference (UI-ready design)
🧪 How It Works
1️⃣ Image Preprocessing

Image resizing to 128×128

Normalization

Data augmentation (rotation, flipping, zooming)

2️⃣ Model Training

Pretrained VGG16 as feature extractor

Custom dense layers for binary classification

Optimized using Adam optimizer and binary crossentropy loss

3️⃣ Evaluation

Metrics: Accuracy, Precision, Recall, F1-score

Validation-based performance monitoring

4️⃣ Prediction

Input image is processed and classified as:

Benign

Malignant

Confidence score included in predictions

🧠 Model Details
Architecture

Base Model: VGG16 (Transfer Learning)

Input Size: 128 × 128 × 3

Classification Type: Binary (Benign vs Malignant)

Final Activation: Sigmoid

Model Summary

Total Parameters: 16,812,355

Trainable Parameters: 9,177,089

Non-Trainable Parameters: 7,635,264

Training Configuration

Optimizer: Adam

Loss Function: Binary Crossentropy

Regularization: Dropout (0.5)

Fine-tuning enabled for upper convolutional layers

Dataset Information

Dataset Size: 1000 images

Training Accuracy: 92%

Validation Accuracy: 88%

📊 Model Performance
Metric	Score
Accuracy	92%
Precision	90%
Recall	93%
F1 Score	91%

⚡ Performance achieved after fine-tuning with dropout and data augmentation on the validation set.

🧾 How to Run the Project
git clone https://github.com/kavyasingh06/SkinCancer.git
cd SkinCancer
pip install -r requirements.txt
python Day3_vgg16_training.py
python Day6_inference.py

📦 Model Files

Due to GitHub file size limitations, trained model files are not stored directly in the repository.

You can:

Train the model using the provided training scripts, or

Integrate Git LFS / Hugging Face Hub for model storage in production setups

💡 Future Enhancements

Integrate Explainable AI (Grad-CAM) for visual interpretability

Deploy using Docker / Hugging Face Spaces

Experiment with advanced architectures (EfficientNet, ResNet50)

Extend to multi-class skin lesion classification

⚠️ Disclaimer

This project is intended for educational and research purposes only.
It is not a medical diagnostic tool and should not be used for clinical decision-making.
Always consult a certified dermatologist for medical concerns.
