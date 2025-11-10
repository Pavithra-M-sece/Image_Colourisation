"""
Quick training script for testing - uses simplified approach
Run this for a faster training session with basic model
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Simple configuration
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20

def load_images_simple(data_dir, max_images=1000):
    """Load images with simple preprocessing"""
    color_dir = os.path.join(data_dir, 'train_color')
    
    if not os.path.exists(color_dir):
        print(f"Directory {color_dir} not found!")
        return None, None
    
    grays = []
    colors = []
    
    image_files = [f for f in os.listdir(color_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:max_images]
    
    print(f"Loading {len(image_files)} images...")
    
    for i, filename in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processed {i}/{len(image_files)} images")
            
        try:
            img_path = os.path.join(color_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = gray.reshape(IMG_SIZE, IMG_SIZE, 1)
            
            grays.append(gray / 255.0)
            colors.append(img / 255.0)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return np.array(grays), np.array(colors)

def build_simple_model():
    """Build a simpler but effective model"""
    
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder
    conv1 = Conv2D(64, (3,3), padding='same', activation='relu')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3,3), padding='same', activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (3,3), padding='same', activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3,3), padding='same', activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, (3,3), padding='same', activation='relu')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3,3), padding='same', activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, (3,3), padding='same', activation='relu')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, (3,3), padding='same', activation='relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottleneck
    conv5 = Conv2D(1024, (3,3), padding='same', activation='relu')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, (3,3), padding='same', activation='relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.5)(conv5)
    
    # Decoder
    up6 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(512, (3,3), padding='same', activation='relu')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3,3), padding='same', activation='relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(256, (3,3), padding='same', activation='relu')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3,3), padding='same', activation='relu')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    up8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(128, (3,3), padding='same', activation='relu')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, (3,3), padding='same', activation='relu')(conv8)
    conv8 = BatchNormalization()(conv8)
    
    up9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = Conv2D(64, (3,3), padding='same', activation='relu')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, (3,3), padding='same', activation='relu')(conv9)
    conv9 = BatchNormalization()(conv9)
    
    # Output
    output = Conv2D(3, (1,1), activation='sigmoid')(conv9)
    
    model = Model(inputs=input_img, outputs=output)
    
    return model

def quick_train():
    """Quick training function"""
    
    print("Loading data...")
    data_dir = './archive (6)/data'
    grays, colors = load_images_simple(data_dir, max_images=1000)
    
    if grays is None:
        print("Failed to load data!")
        return
    
    print(f"Loaded {len(grays)} image pairs")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        grays, colors, test_size=0.2, random_state=42
    )
    
    print("Building model...")
    model = build_simple_model()
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    print(model.summary())
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        ModelCheckpoint('quick_model.keras', save_best_only=True)
    ]
    
    # Train
    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.save('colorization_model.keras')
    print("Model saved as colorization_model.keras")
    
    # Test model
    print("Testing model...")
    test_predictions = model.predict(X_val[:5])
    
    plt.figure(figsize=(15, 10))
    for i in range(5):
        # Original grayscale
        plt.subplot(3, 5, i + 1)
        plt.imshow(X_val[i].squeeze(), cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        # Predicted
        plt.subplot(3, 5, i + 6)
        plt.imshow(test_predictions[i])
        plt.title('AI Colorized')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(3, 5, i + 11)
        plt.imshow(y_val[i])
        plt.title('Original')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, history

if __name__ == "__main__":
    model, history = quick_train()