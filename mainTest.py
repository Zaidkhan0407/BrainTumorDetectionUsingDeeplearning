import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os

model_path = 'BrainTumor.h5'
image_path = 'C:/Users/HP/Desktop/Brain/BrainTumor Classification DL/brain_tumor_dataset/yes/y2.jpg'

model = load_model(model_path)

image = cv2.imread(image_path)
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)
input_img = np.expand_dims(img, axis=0)
input_img = input_img / 255.0


predictions = model.predict(input_img)

threshold = 0.5

if predictions[0][0] > threshold:
    print('Prediction: Tumor present')
else:
    print('Prediction: Tumor not present')