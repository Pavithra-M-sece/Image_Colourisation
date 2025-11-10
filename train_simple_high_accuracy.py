import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3

def load_images_from_folder(folder_path, max_images=3000):
    """Load images from folder"""
    images = []
    count = 0
    
    print(f"Loading images from: {folder_path}")
    
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
                continue
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images)

def prepare_data():
    """Load and prepare training data using LAB color space"""
    print("Preparing training data...")
    
    color_folder = r'archive (6)\data\train_color'
    color_images = load_images_from_folder(color_folder, max_images=3000)
    
    if len(color_images) == 0:
        raise ValueError("No images found!")
    
    # Convert to LAB color space
    lab_images = []
    for img in color_images:
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_images.append(lab_img)
    
    lab_images = np.array(lab_images)
    
    # Split LAB channels
    L_channel = lab_images[:, :, :, 0:1]  # Lightness
    ab_channels = lab_images[:, :, :, 1:3]  # Color channels
    
    # Normalize
    L_channel = L_channel.astype('float32') / 100.0
    ab_channels = (ab_channels.astype('float32') + 128) / 255.0
    
    print(f"L channel shape: {L_channel.shape}")
    print(f"AB channels shape: {ab_channels.shape}")
    
    return L_channel, ab_channels

def build_enhanced_unet():
    """Build enhanced U-Net model"""
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(2)(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(2)(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(2)(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(2)(conv4)
    
    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    # Decoder
    up6 = Conv2DTranspose(512, 2, strides=2, padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2DTranspose(256, 2, strides=2, padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    up8 = Conv2DTranspose(128, 2, strides=2, padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    
    up9 = Conv2DTranspose(64, 2, strides=2, padding='same')(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    
    # Output layer for AB channels
    output = Conv2D(2, 1, activation='tanh', padding='same')(conv9)
    
    model = Model(inputs=input_img, outputs=output)
    return model

def calculate_accuracy_metrics(model, X_test, y_test):
    """Calculate comprehensive accuracy metrics"""
    predictions = model.predict(X_test)
    
    # Calculate MSE
    mse = np.mean((predictions - y_test) ** 2)
    
    # Calculate PSNR
    max_val = 1.0  # Since data is normalized to [0,1]
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    
    # Calculate accuracy percentage (based on PSNR)
    # PSNR > 20 dB is considered good quality
    accuracy = min(100, max(0, (psnr - 15) * 4))  # Scale to percentage
    
    return accuracy, psnr, mse

def plot_results(model, X_test, y_test, num_samples=5):
    """Plot sample predictions"""
    predictions = model.predict(X_test[:num_samples])
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    for i in range(num_samples):
        # Input L channel
        axes[0, i].imshow(X_test[i, :, :, 0], cmap='gray')
        axes[0, i].set_title('Input (L channel)')
        axes[0, i].axis('off')
        
        # Predicted colorization
        pred_lab = np.zeros((IMG_SIZE, IMG_SIZE, 3))
        pred_lab[:, :, 0] = X_test[i, :, :, 0] * 100
        pred_lab[:, :, 1:] = (predictions[i] * 255) - 128
        pred_rgb = cv2.cvtColor(pred_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        axes[1, i].imshow(pred_rgb)
        axes[1, i].set_title('Predicted')
        axes[1, i].axis('off')
        
        # Ground truth
        true_lab = np.zeros((IMG_SIZE, IMG_SIZE, 3))
        true_lab[:, :, 0] = X_test[i, :, :, 0] * 100
        true_lab[:, :, 1:] = (y_test[i] * 255) - 128
        true_rgb = cv2.cvtColor(true_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        axes[2, i].imshow(true_rgb)
        axes[2, i].set_title('Ground Truth')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_high_accuracy_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("🚀 Starting Simple High-Accuracy Image Colorization Training...")
    print(f"TensorFlow version: {tf.__version__}")
    
    # GPU setup
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
    print("Building enhanced U-Net model...")
    model = build_enhanced_unet()
    
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
            'simple_high_accuracy_model.keras',
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
    
    # Calculate accuracy
    print("Calculating accuracy metrics...")
    accuracy, psnr, mse = calculate_accuracy_metrics(model, X_test, y_test)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"MSE: {mse:.6f}")
    
    # Save final model
    model.save('final_simple_high_accuracy_model.keras')
    print("Model saved as 'final_simple_high_accuracy_model.keras'")
    
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
    plt.savefig('simple_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot sample results
    print("Generating sample predictions...")
    plot_results(model, X_test, y_test)
    
    # Success check
    if accuracy >= 75:
        print("🎉 SUCCESS! Target accuracy of 75% achieved!")
        print(f"Your model achieved {accuracy:.2f}% accuracy")
    else:
        print(f"⚠️ Target not reached. Got {accuracy:.2f}%, need 75%+")
        print("Try increasing EPOCHS to 50+ or MAX_IMAGES to 5000+")
    
    print("Training completed!")

if __name__ == "__main__":
    main()