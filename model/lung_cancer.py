import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image

# ---------- Paths ----------
train_dir = 'data/lung_cancer_data/Train_cases'
test_dir = 'data/lung_cancer_data/Test_cases'

# ---------- Parameters ----------
IMG_SIZE = 224  # Increase if you have more compute power
BATCH_SIZE = 32
EPOCHS = 50

# ---------- Data Generators for Train/Val Splitting ----------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.18,  # ~82/18 split for train/val
    rotation_range=18,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.08,
    zoom_range=0.18,
    fill_mode='nearest',
    horizontal_flip=True
)
train_flow = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_flow = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# ---------- CNN Architecture ----------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')  # 3 classes: benign, malignant, normal
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- Training Callbacks ----------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True),
    ModelCheckpoint('lung_cancer_best.h5', save_best_only=True, monitor='val_accuracy')
]

# ---------- Training ----------
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------- Save Model for Deployment ----------
model.save('lung_cancer_final.h5')  # Use this .h5 file for inference/deployment

# ---------- Predict on Test Images (no subfolders) ----------
import glob

test_imgs = glob.glob(os.path.join(test_dir, '*'))  # All file paths in test_cases
class_indices = {v: k for k, v in train_flow.class_indices.items()}  # Map numeric -> class

for img_path in test_imgs:
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred_class = np.argmax(pred)
    print(f"{os.path.basename(img_path)}: {class_indices[pred_class]}")
