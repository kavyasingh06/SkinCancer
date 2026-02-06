import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------
# Paths setup
# ----------------------------
BASE_DIR = "data"
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "skin_cancer_vgg16_finetuned.h5")

# ----------------------------
# Load model
# ----------------------------
model = load_model(MODEL_PATH)
print(f"✅ Loaded model from: {MODEL_PATH}")

# ----------------------------
# Validation generator
# ----------------------------
val_datagen = ImageDataGenerator(rescale=1./255)  # ✅ FIXED

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Check data distribution
print("Class distribution in validation data:", dict(zip(val_gen.class_indices.keys(), np.bincount(val_gen.classes))))

# ----------------------------
# Evaluate
# ----------------------------
loss, accuracy = model.evaluate(val_gen)
print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# ----------------------------
# Predictions & Confusion Matrix
# ----------------------------
pred_probs = model.predict(val_gen)
pred_classes = (pred_probs > 0.5).astype(int).reshape(-1)
true_classes = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

report = classification_report(true_classes, pred_classes, target_names=class_labels)
print("\nClassification Report:\n", report)
