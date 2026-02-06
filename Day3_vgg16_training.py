
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

# ----------------------------
# Paths setup
# ----------------------------
BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "skin_cancer_vgg16_finetuned.h5")
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# Data Generators
# ----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# ----------------------------
# Compute class weights
# ----------------------------
classes = train_gen.classes
class_weights_values = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)
class_weights = dict(enumerate(class_weights_values))
print("Class weights:", class_weights)

# ----------------------------
# Load VGG16 base model
# ----------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    layer.trainable = False

# ----------------------------
# Add custom top layers
# ----------------------------
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ----------------------------
# Callbacks
# ----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')

# ----------------------------
# Train model (initial)
# ----------------------------
EPOCHS = 10
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights
)

# ----------------------------
# Fine-tune last VGG16 layers
# ----------------------------
for layer in base_model.layers[15:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

EPOCHS_FINE = 10
history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights
)

# ----------------------------
# Save final model
# ----------------------------
model.save(MODEL_PATH)
print(f"✅ Fine-tuned VGG16 model saved at: {MODEL_PATH}")
