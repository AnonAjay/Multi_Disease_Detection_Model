import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#--- Main Folder Containing Class Subfolders ---#
data_dir = 'data/gastro_intestinal_diseases_data'  # Change this to your dataset folder

#--- Parameters ---#
IMG_SIZE = 224      # For speed. Use 224/256 for bigger images if you have GPU.
BATCH_SIZE = 32
EPOCHS = 60

#--- Data Generator (uses validation_split for automatic train/val splitting) ---#
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.18,  # About 82/18 train/val split
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

train_flow = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_flow = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

#--- CNN Model ---#
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
    Dense(8, activation='softmax')   # 8 classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#--- Callbacks: Best Model Saved for Deployment ---#
callbacks = [
    EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True),
    ModelCheckpoint('gastro_cnn_best.h5', save_best_only=True, monitor='val_accuracy')
]

#--- Training ---#
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    callbacks=callbacks
)

#--- Save final model for deploy (architecture + weights) ---#
model.save('gastro_cnn_final.h5')  # This .h5 file can be loaded for deployment

#--- To predict on new images later: ---
# from tensorflow.keras.models import load_model
# loaded_model = load_model('gastro_cnn_final.h5')
# # Load and preprocess new image as 'img_array'
# pred = loaded_model.predict(img_array)  # Output: numpy array of 8 probabilities
# pred_class = np.argmax(pred)
