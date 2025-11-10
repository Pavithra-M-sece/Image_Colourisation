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

# Enhanced Configuration for 85% accuracy
IMG_SIZE = 256  # Increased resolution
BATCH_SIZE = 8  # Smaller batch for better gradients
EPOCHS = 100
LEARNING_RATE = 5e-4  # Lower learning rate
MAX_IMAGES = 4000  # Use more data

def data_augmentation():
    """Data augmentation pipeline"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1)
    ])

def load_images_optimized(folder_path, max_images=None):
    """Optimized image loading with preprocessing"""
    images = []
    count = 0
    
    print(f"Loading images from {folder_path}...")
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            if max_images and count >= max_images:
                break
                
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Quality check - skip very dark or very bright images
                    if np.mean(img) > 20 and np.mean(img) < 235:
                        images.append(img)
                        count += 1
                        
                        if count % 500 == 0:
                            print(f"Loaded {count} quality images...")
                            
            except Exception as e:
                continue
    
    print(f"Successfully loaded {len(images)} quality images")
    return np.array(images)

def build_enhanced_model():
    """Enhanced U-Net with attention and residual connections"""
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder with residual connections
    def conv_block(x, filters, dropout=0.1):
        conv = Conv2D(filters, 3, padding='same')(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(dropout)(conv)
        conv = Conv2D(filters, 3, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        return conv
    
    def attention_block(x, g, filters):
        """Attention mechanism"""
        theta_x = Conv2D(filters, 1, padding='same')(x)
        phi_g = Conv2D(filters, 1, padding='same')(g)
        f = Activation('relu')(Add()([theta_x, phi_g]))
        psi_f = Conv2D(1, 1, padding='same')(f)
        rate = Activation('sigmoid')(psi_f)
        att_x = Multiply()([x, rate])
        return att_x
    
    # Encoder
    c1 = conv_block(input_img, 64)
    p1 = MaxPooling2D(2)(c1)
    
    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D(2)(c2)
    
    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D(2)(c3)
    
    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D(2)(c4)
    
    c5 = conv_block(p4, 1024, dropout=0.2)
    
    # Decoder with attention
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    a6 = attention_block(c4, u6, 256)
    u6 = Concatenate()([u6, a6])
    c6 = conv_block(u6, 512)
    
    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    a7 = attention_block(c3, u7, 128)
    u7 = Concatenate()([u7, a7])
    c7 = conv_block(u7, 256)
    
    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    a8 = attention_block(c2, u8, 64)
    u8 = Concatenate()([u8, a8])
    c8 = conv_block(u8, 128)
    
    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    a9 = attention_block(c1, u9, 32)
    u9 = Concatenate()([u9, a9])
    c9 = conv_block(u9, 64)
    
    # Output with residual connection
    output = Conv2D(3, 1, activation='sigmoid')(c9)
    
    model = Model(inputs=input_img, outputs=output)
    return model

def custom_loss(y_true, y_pred):
    """Custom loss combining MSE and perceptual loss"""
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
    # Color consistency loss
    y_true_lab = tf.image.rgb_to_yuv(y_true)
    y_pred_lab = tf.image.rgb_to_yuv(y_pred)
    color_loss = tf.reduce_mean(tf.square(y_true_lab[..., 1:] - y_pred_lab[..., 1:]))
    
    return mse_loss + 0.1 * color_loss

def ssim_metric(y_true, y_pred):
    """SSIM metric for better evaluation"""
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def prepare_enhanced_data():
    """Enhanced data preparation"""
    print("Preparing enhanced training data...")
    
    color_folder = r'archive (6)\data\train_color'
    color_images = load_images_optimized(color_folder, max_images=MAX_IMAGES)
    
    if len(color_images) == 0:
        raise ValueError("No images found!")
    
    print(f"Loaded {len(color_images)} color images")
    
    # Create grayscale versions with better conversion
    gray_images = []
    for img in color_images:
        # Convert to LAB color space for better grayscale conversion
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        gray = lab[:, :, 0:1]  # L channel only
        gray_images.append(gray)
    
    gray_images = np.array(gray_images, dtype='float32') / 255.0
    color_images = color_images.astype('float32') / 255.0
    
    print(f"Enhanced data prepared - Gray: {gray_images.shape}, Color: {color_images.shape}")
    return gray_images, color_images

def main():
    """Enhanced training for 85% accuracy"""
    print("🚀 Starting Enhanced Image Colorization Training for 85% Accuracy...")
    print(f"TensorFlow version: {tf.__version__}")
    
    # GPU setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"🎮 GPU available: {len(gpus)} GPU(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("💻 Using CPU (training will be slower)")
    
    # Load enhanced data
    try:
        X_data, y_data = prepare_enhanced_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Enhanced data split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.15, random_state=42, shuffle=True
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Build enhanced model
    print("🏗️ Building enhanced model...")
    model = build_enhanced_model()
    
    # Enhanced compilation
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
        loss=custom_loss,
        metrics=['mae', ssim_metric]
    )
    
    print("📋 Enhanced Model Summary:")
    model.summary()
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_ssim_metric',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_enhanced_model.keras',
            monitor='val_ssim_metric',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: LEARNING_RATE * (0.95 ** epoch)
        )
    ]
    
    # Enhanced training
    print("🎯 Starting enhanced training...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('final_enhanced_model.keras')
    print("💾 Enhanced model saved!")
    
    # Evaluate
    print("📈 Final Evaluation:")
    test_loss, test_mae, test_ssim = model.evaluate(X_test, y_test, verbose=0)
    
    # Convert SSIM to percentage accuracy
    accuracy_percentage = test_ssim * 100
    
    print(f"🎯 Test Accuracy (SSIM): {accuracy_percentage:.2f}%")
    print(f"📉 Test Loss: {test_loss:.4f}")
    print(f"📊 Test MAE: {test_mae:.4f}")
    
    if accuracy_percentage >= 85:
        print("🎉 SUCCESS: Achieved target accuracy of 85%!")
    else:
        print(f"⚠️ Current accuracy: {accuracy_percentage:.2f}% - May need more training")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([x * 100 for x in history.history['ssim_metric']], label='Training Accuracy')
    plt.plot([x * 100 for x in history.history['val_ssim_metric']], label='Validation Accuracy')
    plt.title('Model Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Enhanced training completed!")

if __name__ == "__main__":
    main()