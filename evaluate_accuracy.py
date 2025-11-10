import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 25

def load_images_from_folder(folder_path, max_images=2000):
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
    """Load and prepare training data"""
    print("Preparing training data...")
    
    color_folder = r'archive (6)\data\train_color'
    color_images = load_images_from_folder(color_folder, max_images=2000)
    
    if len(color_images) == 0:
        raise ValueError("No images found!")
    
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
    
    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    
    # Decoder
    up5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4)
    up5 = concatenate([up5, conv3])
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv2])
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv1])
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    output = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(conv7)
    
    model = Model(inputs=input_img, outputs=output)
    return model

def calculate_psnr(y_true, y_pred):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(y_true, y_pred):
    """Calculate SSIM (Structural Similarity Index)"""
    def ssim_single(img1, img2):
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2
        
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        return ssim_val
    
    ssim_values = []
    for i in range(len(y_true)):
        ssim_val = ssim_single(y_true[i], y_pred[i])
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def evaluate_model_accuracy(model, X_test, y_test):
    """Calculate comprehensive accuracy metrics"""
    print("Evaluating model accuracy...")
    
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    
    # Calculate MSE
    mse = np.mean((predictions - y_test) ** 2)
    
    # Calculate MAE
    mae = np.mean(np.abs(predictions - y_test))
    
    # Calculate PSNR
    psnr = calculate_psnr(y_test, predictions)
    
    # Calculate SSIM
    ssim = calculate_ssim(y_test, predictions)
    
    # Calculate accuracy percentage based on PSNR
    # PSNR > 20 dB is considered good quality
    # PSNR > 25 dB is considered very good quality
    # PSNR > 30 dB is considered excellent quality
    if psnr >= 30:
        accuracy = 95 + min(5, (psnr - 30) * 0.5)  # 95-100%
    elif psnr >= 25:
        accuracy = 85 + (psnr - 25) * 2  # 85-95%
    elif psnr >= 20:
        accuracy = 70 + (psnr - 20) * 3  # 70-85%
    elif psnr >= 15:
        accuracy = 50 + (psnr - 15) * 4  # 50-70%
    else:
        accuracy = max(0, psnr * 3.33)  # 0-50%
    
    return {
        'accuracy_percentage': accuracy,
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse,
        'mae': mae
    }

def main():
    """Main evaluation function"""
    print("🎯 AI Image Colorization - Accuracy Evaluation")
    print("=" * 50)
    
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
    
    # Build and compile model
    print("Building model...")
    model = build_colorization_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    print("Training model...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate accuracy
    metrics = evaluate_model_accuracy(model, X_test, y_test)
    
    # Print results
    print("\n" + "=" * 50)
    print("🎯 FINAL ACCURACY RESULTS")
    print("=" * 50)
    print(f"📊 Overall Accuracy: {metrics['accuracy_percentage']:.2f}%")
    print(f"📈 PSNR (Peak Signal-to-Noise Ratio): {metrics['psnr']:.2f} dB")
    print(f"🔍 SSIM (Structural Similarity): {metrics['ssim']:.4f}")
    print(f"📉 MSE (Mean Squared Error): {metrics['mse']:.6f}")
    print(f"📏 MAE (Mean Absolute Error): {metrics['mae']:.6f}")
    print(f"⏱️ Training Time: {training_time:.2f} seconds")
    print("=" * 50)
    
    # Quality assessment
    if metrics['accuracy_percentage'] >= 85:
        quality = "🏆 EXCELLENT"
    elif metrics['accuracy_percentage'] >= 75:
        quality = "🥇 VERY GOOD"
    elif metrics['accuracy_percentage'] >= 65:
        quality = "🥈 GOOD"
    elif metrics['accuracy_percentage'] >= 50:
        quality = "🥉 FAIR"
    else:
        quality = "❌ NEEDS IMPROVEMENT"
    
    print(f"Quality Rating: {quality}")
    
    # Save model
    model.save('accuracy_evaluated_model.keras')
    print("Model saved as 'accuracy_evaluated_model.keras'")
    
    return metrics

if __name__ == "__main__":
    results = main()