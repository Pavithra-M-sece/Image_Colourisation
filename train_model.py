import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, concatenate, BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import time

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3

def load_images_from_folder(folder_path, max_images=None):
    """Load images from a folder and return as numpy array"""
    images = []
    count = 0
    
    print(f"Loading images from {folder_path}...")
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            if max_images and count >= max_images:
                break
                
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
                    count += 1
                    
                    if count % 500 == 0:
                        print(f"Loaded {count} images...")
                        
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images)

def prepare_data():
    """Load and prepare training data"""
    print("Preparing training data...")
    
    # Load color images
    color_folder = r'archive (6)\data\train_color'
    
    # Load a subset for faster training (you can increase this)
    color_images = load_images_from_folder(color_folder, max_images=2000)
    
    if len(color_images) == 0:
        raise ValueError("No images found! Check the folder path.")
    
    print(f"Loaded {len(color_images)} color images")
    
    # Create grayscale versions
    gray_images = []
    for img in color_images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray.reshape(IMG_SIZE, IMG_SIZE, 1)
        gray_images.append(gray)
    
    gray_images = np.array(gray_images)
    
    # Normalize to [0, 1]
    gray_images = gray_images.astype('float32') / 255.0
    color_images = color_images.astype('float32') / 255.0
    
    print(f"Gray images shape: {gray_images.shape}")
    print(f"Color images shape: {color_images.shape}")
    
    return gray_images, color_images

def build_colorization_model():
    """Build U-Net style colorization model"""
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder (downsampling path)
    # Block 1
    conv1 = Conv2D(64, (3, 3), padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    
    # Block 2
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    
    # Block 3
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    
    # Block 4
    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    
    # Decoder (upsampling path)
    # Block 5
    up5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4)
    up5 = concatenate([up5, conv3])
    conv5 = Conv2D(256, (3, 3), padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(256, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    # Block 6
    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv2])
    conv6 = Conv2D(128, (3, 3), padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(128, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    
    # Block 7
    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv1])
    conv7 = Conv2D(64, (3, 3), padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(64, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    
    # Output layer
    output = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(conv7)
    
    model = Model(inputs=input_img, outputs=output)
    
    return model

def plot_sample_predictions(model, X_test, y_test, num_samples=5):
    """Plot sample predictions"""
    predictions = model.predict(X_test[:num_samples])
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    for i in range(num_samples):
        # Original grayscale
        axes[0, i].imshow(X_test[i].squeeze(), cmap='gray')
        axes[0, i].set_title('Grayscale Input')
        axes[0, i].axis('off')
        
        # Predicted colorization
        axes[1, i].imshow(predictions[i])
        axes[1, i].set_title('Predicted')
        axes[1, i].axis('off')
        
        # Ground truth
        axes[2, i].imshow(y_test[i])
        axes[2, i].set_title('Ground Truth')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("Starting Image Colorization Training...")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} GPU(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU")
    
    # Load and prepare data
    try:
        X_data, y_data = prepare_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build model
    print("Building model...")
    model = build_colorization_model()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    print("Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_colorization_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    model.save('final_colorization_model.keras')
    print("Model saved as 'final_colorization_model.keras'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot sample predictions
    print("Generating sample predictions...")
    plot_sample_predictions(model, X_test, y_test)
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()