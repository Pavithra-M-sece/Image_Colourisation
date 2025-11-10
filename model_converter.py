"""
Script to convert and save the trained model from the Colab notebook
Run this script to save your trained model in the correct format
"""

import tensorflow as tf
import numpy as np
import os

def build_colorization_model():
    """Build the same model architecture as in the notebook"""
    IMG_SIZE = 128
    
    input_img = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Encoder: downsampling + feature extraction
    conv1 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_img)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv2 = tf.keras.layers.Conv2D(128, (3,3), padding='same', strides=2, activation='relu')(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)

    # Bottleneck feature extraction
    conv3 = tf.keras.layers.Conv2D(256, (3,3), padding='same', strides=2, activation='relu')(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv4 = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu')(conv3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)

    # Decoder: upsampling and concatenation
    up1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=1, padding='same', activation='relu')(conv4)
    up1 = tf.keras.layers.BatchNormalization()(up1)
    up1 = tf.keras.layers.concatenate([up1, conv3])
    conv5 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(up1)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)

    up2 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(conv5)
    up2 = tf.keras.layers.BatchNormalization()(up2)
    up2 = tf.keras.layers.concatenate([up2, conv2])
    conv6 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(up2)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)

    # Upsample to original image size (128x128)
    up3 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(conv6)
    up3 = tf.keras.layers.BatchNormalization()(up3)

    # Final color prediction layer
    conv7 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(up3)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    output_img = tf.keras.layers.Conv2D(3, (1,1), activation='sigmoid', padding='same')(conv7)

    model = tf.keras.Model(input_img, output_img)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse',
                  metrics=['accuracy'])
    return model

def convert_model():
    """Convert the model from Google Drive to local format"""
    
    # Check if the original model exists
    drive_model_path = '/content/drive/My Drive/colorization_model.keras'
    local_model_path = 'colorization_model.keras'
    
    if os.path.exists(drive_model_path):
        print("Loading model from Google Drive...")
        try:
            model = tf.keras.models.load_model(drive_model_path)
            print("Model loaded successfully!")
            
            # Save to local directory
            model.save(local_model_path)
            print(f"Model saved to {local_model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new model with the same architecture...")
            model = build_colorization_model()
            model.save(local_model_path)
            print("New model created and saved!")
    else:
        print("Google Drive model not found. Creating new model...")
        model = build_colorization_model()
        model.save(local_model_path)
        print("New model created and saved!")

if __name__ == "__main__":
    convert_model()