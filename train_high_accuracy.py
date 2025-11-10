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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2

def load_images_from_folder(folder_path, max_images=None):
    """Load images from folder with error handling"""
    images = []
    count = 0
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return np.array([])
    
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
                print(f"Error loading {filename}: {e}")
                continue
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images)

def rgb_to_lab(rgb_images):
    """Convert RGB to LAB color space"""
    lab_images = []
    for rgb_img in rgb_images:
        # Convert to LAB
        lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        lab_images.append(lab_img)
    return np.array(lab_images)

def prepare_data():
    """Load and prepare training data with LAB color space"""
    print("Preparing training data...")
    
    # Load color images
    color_folder = r'archive (6)\data\train_color'
    
    # Load more images for better training
    color_images = load_images_from_folder(color_folder, max_images=4000)
    
    if len(color_images) == 0:
        raise ValueError("No images found! Check the folder path.")
    
    print(f"Loaded {len(color_images)} color images")
    
    # Convert to LAB color space for better colorization
    lab_images = rgb_to_lab(color_images)
    
    # Split LAB channels
    L_channel = lab_images[:, :, :, 0:1]  # Lightness (grayscale)
    ab_channels = lab_images[:, :, :, 1:3]  # Color channels
    
    # Normalize
    L_channel = L_channel.astype('float32') / 100.0  # L channel range [0, 100]
    ab_channels = (ab_channels.astype('float32') + 128) / 255.0  # AB channels range [-128, 127]
    
    print(f"L channel shape: {L_channel.shape}")
    print(f"AB channels shape: {ab_channels.shape}")
    
    return L_channel, ab_channels

def build_improved_unet():
    """Build improved U-Net with attention and residual connections"""
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder with residual connections
    def conv_block(x, filters, dropout=0.1):
        conv = Conv2D(filters, 3, padding='same')(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(dropout)(conv)
        conv = Conv2D(filters, 3, padding='same')(conv)
        conv = BatchNormalization()(conv)
        return conv
    
    def residual_block(x, filters):
        shortcut = x
        if x.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, padding='same')(x)
        
        conv = conv_block(x, filters)
        conv = Add()([conv, shortcut])
        conv = Activation('relu')(conv)
        return conv
    
    # Encoder
    conv1 = residual_block(input_img, 64)
    pool1 = MaxPooling2D(2)(conv1)
    
    conv2 = residual_block(pool1, 128)
    pool2 = MaxPooling2D(2)(conv2)
    
    conv3 = residual_block(pool2, 256)
    pool3 = MaxPooling2D(2)(conv3)
    
    conv4 = residual_block(pool3, 512)
    pool4 = MaxPooling2D(2)(conv4)
    
    # Bottleneck with attention
    conv5 = residual_block(pool4, 1024)
    
    # Attention mechanism
    attention = GlobalAveragePooling2D()(conv5)
    attention = Dense(1024 // 16, activation='relu')(attention)
    attention = Dense(1024, activation='sigmoid')(attention)
    attention = Reshape((1, 1, 1024))(attention)
    conv5 = Multiply()([conv5, attention])
    
    # Decoder with skip connections
    up6 = Conv2DTranspose(512, 2, strides=2, padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = residual_block(up6, 512)
    
    up7 = Conv2DTranspose(256, 2, strides=2, padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = residual_block(up7, 256)
    
    up8 = Conv2DTranspose(128, 2, strides=2, padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = residual_block(up8, 128)
    
    up9 = Conv2DTranspose(64, 2, strides=2, padding='same')(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = residual_block(up9, 64)
    
    # Output layer for AB channels
    output = Conv2D(2, 1, activation='tanh', padding='same')(conv9)
    
    model = Model(inputs=input_img, outputs=output)
    return model

def custom_loss(y_true, y_pred):
    """Custom loss combining MSE and perceptual loss"""
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
    # Add gradient loss for better edge preservation
    def gradient_loss(y_true, y_pred):
        def gradient(x):
            h_x = x.shape[1]
            w_x = x.shape[2]
            r = x[:, 1:h_x, :, :]
            l = x[:, 0:h_x-1, :, :]
            t = x[:, :, 1:w_x, :]
            b = x[:, :, 0:w_x-1, :]
            dx = tf.abs(r - l)
            dy = tf.abs(t - b)
            return dx, dy
        
        dx_true, dy_true = gradient(y_true)
        dx_pred, dy_pred = gradient(y_pred)
        
        grad_loss = tf.reduce_mean(tf.abs(dx_true - dx_pred)) + tf.reduce_mean(tf.abs(dy_true - dy_pred))
        return grad_loss
    
    grad_loss = gradient_loss(y_true, y_pred)
    
    return mse_loss + 0.1 * grad_loss

def data_generator(X, y, batch_size=32, augment=True):
    """Data generator with augmentation"""
    while True:
        indices = np.random.permutation(len(X))
        
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X[batch_indices].copy()
            batch_y = y[batch_indices].copy()
            
            if augment:
                # Random horizontal flip
                for j in range(len(batch_X)):
                    if np.random.random() > 0.5:
                        batch_X[j] = np.fliplr(batch_X[j])
                        batch_y[j] = np.fliplr(batch_y[j])
            
            yield batch_X, batch_y

def calculate_accuracy(model, X_test, y_test):
    """Calculate colorization accuracy using PSNR and SSIM"""
    predictions = model.predict(X_test)
    
    psnr_scores = []
    ssim_scores = []
    
    for i in range(len(predictions)):
        # Convert back to RGB for evaluation
        pred_lab = np.zeros((IMG_SIZE, IMG_SIZE, 3))
        true_lab = np.zeros((IMG_SIZE, IMG_SIZE, 3))
        
        pred_lab[:, :, 0] = X_test[i, :, :, 0] * 100
        pred_lab[:, :, 1:] = (predictions[i] * 255) - 128
        
        true_lab[:, :, 0] = X_test[i, :, :, 0] * 100
        true_lab[:, :, 1:] = (y_test[i] * 255) - 128
        
        # Convert to RGB
        pred_rgb = cv2.cvtColor(pred_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        true_rgb = cv2.cvtColor(true_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # Calculate PSNR
        mse = np.mean((pred_rgb - true_rgb) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        psnr_scores.append(psnr)
        
        # Calculate SSIM (simplified version)
        def ssim(img1, img2):
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1 = np.var(img1)
            sigma2 = np.var(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
            return ssim_val
        
        ssim_score = ssim(pred_rgb, true_rgb)
        ssim_scores.append(ssim_score)
    
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    # Convert to percentage accuracy (PSNR > 20 is considered good)
    accuracy = min(100, max(0, (avg_psnr - 10) * 5))  # Scale PSNR to percentage
    
    return accuracy, avg_psnr, avg_ssim

def plot_results(model, X_test, y_test, num_samples=5):
    """Plot sample predictions"""
    predictions = model.predict(X_test[:num_samples])
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    for i in range(num_samples):
        # Convert back to RGB for visualization
        # Grayscale input
        gray_img = X_test[i, :, :, 0]
        axes[0, i].imshow(gray_img, cmap='gray')
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
    plt.savefig('high_accuracy_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("Starting High-Accuracy Image Colorization Training...")
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
        X_data, y_data, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build model
    print("Building improved U-Net model...")
    model = build_improved_unet()
    
    # Compile with custom loss
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=custom_loss,
        metrics=['mae']
    )
    
    print("Model Summary:")
    model.summary()
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'high_accuracy_colorization_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Custom callback to track accuracy
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1} - Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")
        )
    ]
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    
    # Use data generator for augmentation
    train_gen = data_generator(X_train, y_train, BATCH_SIZE, augment=True)
    val_gen = data_generator(X_test, y_test, BATCH_SIZE, augment=False)
    
    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_test) // BATCH_SIZE
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Calculate final accuracy
    print("Calculating accuracy...")
    accuracy, psnr, ssim = calculate_accuracy(model, X_test, y_test)
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Average PSNR: {psnr:.2f} dB")
    print(f"Average SSIM: {ssim:.4f}")
    
    # Save final model
    model.save('final_high_accuracy_model.keras')
    print("Model saved as 'final_high_accuracy_model.keras'")
    
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
    plt.savefig('high_accuracy_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot sample results
    print("Generating sample predictions...")
    plot_results(model, X_test, y_test)
    
    print(f"Training completed successfully with {accuracy:.2f}% accuracy!")
    
    if accuracy >= 75:
        print("🎉 Target accuracy of 75% achieved!")
    else:
        print("⚠️ Target accuracy not reached. Consider:")
        print("- Increasing training epochs")
        print("- Adding more training data")
        print("- Fine-tuning hyperparameters")

if __name__ == "__main__":
    main()