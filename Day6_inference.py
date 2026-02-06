import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = os.path.join("models", "skin_cancer_vgg16_finetuned.h5")
IMAGE_PATH = "C:/Users/kavya/Downloads/S_1017_melanotic_malignant_melanoma_M1310297.width-1534.jpg"

# ----------------------------
# Load model
# ----------------------------
model = load_model(MODEL_PATH)
print(f"✅ Loaded model from {MODEL_PATH}")

# ----------------------------
# Load & preprocess image
# ----------------------------
img = Image.open(IMAGE_PATH).resize((224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# ----------------------------
# Predict
# ----------------------------
pred_prob = model.predict(img_array)[0][0]
pred_class = "Malignant" if pred_prob > 0.5 else "Benign"
print(f"Prediction: {pred_class} ({pred_prob:.4f})")

# ----------------------------
# Show image
# ----------------------------
plt.imshow(img)
plt.title(f"Predicted: {pred_class} ({pred_prob:.2f})")
plt.axis("off")
plt.show()
