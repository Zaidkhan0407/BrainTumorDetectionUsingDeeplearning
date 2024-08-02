import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# Define and train the autoencoder model
autoencoder = Sequential()

# Encoder layers
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))

# Decoder layers
autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Load the trained autoencoder weights
autoencoder.load_weights('autoencoder_model.h5')

# Preprocess your image data
image_directory = 'C:/Users/HP/Desktop/Brain/BrainTumor Classification DL/brain_tumor_dataset/'

no_tumor_images = os.listdir(os.path.join(image_directory, 'no/'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes/'))

preprocessed_images = []

for image_name in no_tumor_images:
    image_path = os.path.join(image_directory, 'no', image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize image to 64x64 pixels
    image = image / 255.0  # Normalize pixel values to [0, 1]
    preprocessed_images.append(image)

for image_name in yes_tumor_images:
    image_path = os.path.join(image_directory, 'yes', image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize image to 64x64 pixels
    image = image / 255.0  # Normalize pixel values to [0, 1]
    preprocessed_images.append(image)

preprocessed_images = np.array(preprocessed_images)

# Use the autoencoder to reconstruct the images
reconstructed_images = autoencoder.predict(preprocessed_images)

# Now `reconstructed_images` contains the reconstructed images

# Pass the reconstructed images to your existing brain tumor classification model
# for further processing
