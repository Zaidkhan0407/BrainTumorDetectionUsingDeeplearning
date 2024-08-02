import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


# Define input shape (assuming images are 64x64 pixels with 3 channels for RGB)
input_shape = (64, 64, 3)

# Define the encoder part of the autoencoder
encoder_input = Input(shape=input_shape)
encoder_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
encoder_pool1 = MaxPooling2D((2, 2), padding='same')(encoder_conv1)
encoder_conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_pool1)
encoder_pool2 = MaxPooling2D((2, 2), padding='same')(encoder_conv2)
encoder_conv3 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_pool2)
encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_conv3)

# Define the decoder part of the autoencoder
decoder_conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_output)
decoder_upsamp1 = UpSampling2D((2, 2))(decoder_conv1)
decoder_conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_upsamp1)
decoder_upsamp2 = UpSampling2D((2, 2))(decoder_conv2)
decoder_conv3 = Conv2D(32, (3, 3), activation='relu')(decoder_upsamp2)
decoder_upsamp3 = UpSampling2D((2, 2))(decoder_conv3)
decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder_upsamp3)

# Create the autoencoder model
autoencoder = Model(encoder_input, decoder_output)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Print model summary
autoencoder.summary()
autoencoder.save('autoencoder_model.h5')
