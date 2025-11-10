import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import albumentations as A

# Configuration
IMG_SIZE = 256  # Increased from 128 for better quality
BATCH_SIZE = 8  # Reduced for memory efficiency
EPOCHS = 50
LEARNING_RATE = 1e-4

def create_data_augmentation():
    """Create data augmentation pipeline"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
    ])

def load_and_preprocess_data(data_dir, max_images=None):
    """Load and preprocess training data with augmentation"""
    color_dir = os.path.join(data_dir, 'train_color')
    
    if not os.path.exists(color_dir):
        print(f"Directory {color_dir} not found!")
        return None, None
    
    images = []
    grays = []
    colors = []
    
    # Get list of image files
    image_files = [f for f in os.listdir(color_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Loading {len(image_files)} images...")
    
    augment = create_data_augmentation()
    
    for i, filename in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processed {i}/{len(image_files)} images")
            
        try:
            # Load image
            img_path = os.path.join(color_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Apply augmentation
            augmented = augment(image=img)
            img_aug = augmented['image']
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_aug, cv2.COLOR_RGB2GRAY)
            gray = gray.reshape(IMG_SIZE, IMG_SIZE, 1)
            
            # Normalize
            gray_norm = gray / 255.0
            color_norm = img_aug / 255.0
            
            grays.append(gray_norm)
            colors.append(color_norm)
            
            # Add original image without augmentation
            gray_orig = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_orig = gray_orig.reshape(IMG_SIZE, IMG_SIZE, 1)
            gray_orig_norm = gray_orig / 255.0
            color_orig_norm = img / 255.0
            
            grays.append(gray_orig_norm)
            colors.append(color_orig_norm)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return np.array(grays), np.array(colors)

def build_improved_model():
    """Build improved U-Net model with attention and residual connections"""
    
    def conv_block(x, filters, kernel_size=3, strides=1):
        """Convolutional block with batch normalization and activation"""
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def residual_block(x, filters):
        """Residual block"""
        shortcut = x
        
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        
        # Match dimensions if needed
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    def attention_gate(g, x, filters):
        """Attention gate for skip connections"""
        g1 = Conv2D(filters, 1, padding='same')(g)
        g1 = BatchNormalization()(g1)
        
        x1 = Conv2D(filters, 1, padding='same')(x)
        x1 = BatchNormalization()(x1)
        
        psi = Add()([g1, x1])
        psi = Activation('relu')(psi)
        psi = Conv2D(1, 1, padding='same')(psi)
        psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)
        
        return Multiply()([x, psi])
    
    # Input
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder
    e1 = conv_block(inputs, 64)
    e1 = residual_block(e1, 64)
    p1 = MaxPooling2D(2)(e1)
    
    e2 = conv_block(p1, 128)
    e2 = residual_block(e2, 128)
    p2 = MaxPooling2D(2)(e2)
    
    e3 = conv_block(p2, 256)
    e3 = residual_block(e3, 256)
    p3 = MaxPooling2D(2)(e3)
    
    e4 = conv_block(p3, 512)
    e4 = residual_block(e4, 512)
    p4 = MaxPooling2D(2)(e4)
    
    # Bottleneck
    b = conv_block(p4, 1024)
    b = residual_block(b, 1024)
    b = Dropout(0.5)(b)
    
    # Decoder with attention
    u4 = Conv2DTranspose(512, 2, strides=2, padding='same')(b)
    a4 = attention_gate(u4, e4, 256)
    u4 = Concatenate()([u4, a4])
    d4 = conv_block(u4, 512)
    d4 = residual_block(d4, 512)
    
    u3 = Conv2DTranspose(256, 2, strides=2, padding='same')(d4)
    a3 = attention_gate(u3, e3, 128)
    u3 = Concatenate()([u3, a3])
    d3 = conv_block(u3, 256)
    d3 = residual_block(d3, 256)
    
    u2 = Conv2DTranspose(128, 2, strides=2, padding='same')(d3)
    a2 = attention_gate(u2, e2, 64)
    u2 = Concatenate()([u2, a2])
    d2 = conv_block(u2, 128)
    d2 = residual_block(d2, 128)
    
    u1 = Conv2DTranspose(64, 2, strides=2, padding='same')(d2)
    a1 = attention_gate(u1, e1, 32)
    u1 = Concatenate()([u1, a1])
    d1 = conv_block(u1, 64)
    d1 = residual_block(d1, 64)
    
    # Output
    outputs = Conv2D(3, 1, activation='sigmoid', padding='same')(d1)
    
    model = Model(inputs, outputs, name='ImprovedColorization')
    
    return model

def perceptual_loss(y_true, y_pred):
    """Perceptual loss using VGG features"""
    # Load VGG19 model
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    vgg.trainable = False
    
    # Get features from multiple layers
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
    outputs = [vgg.get_layer(name).output for name in layer_names]
    feature_extractor = Model(vgg.input, outputs)
    
    # Extract features
    true_features = feature_extractor(y_true)
    pred_features = feature_extractor(y_pred)
    
    # Calculate loss
    loss = 0
    for true_feat, pred_feat in zip(true_features, pred_features):
        loss += tf.reduce_mean(tf.square(true_feat - pred_feat))
    
    return loss

def combined_loss(y_true, y_pred):
    """Combined loss: MSE + Perceptual + SSIM"""
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # SSIM loss
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    # Perceptual loss (simplified)
    perc_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # Placeholder
    
    return 0.6 * mse_loss + 0.3 * ssim_loss + 0.1 * perc_loss

def train_model():
    """Main training function"""
    print("Loading data...")
    
    # Load data
    data_dir = './archive (6)/data'
    grays, colors = load_and_preprocess_data(data_dir, max_images=2000)  # Limit for demo
    
    if grays is None:
        print("Failed to load data!")
        return
    
    print(f"Loaded {len(grays)} image pairs")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        grays, colors, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Build model
    print("Building model...")
    model = build_improved_model()
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=['mae', 'mse']
    )
    
    print(model.summary())
    
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
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('final_colorization_model.keras')
    print("Model saved!")
    
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
    plt.savefig('training_history.png')
    plt.show()
    
    return model, history

def test_model(model, test_images=5):
    """Test the trained model on sample images"""
    data_dir = './archive (6)/data'
    test_dir = os.path.join(data_dir, 'test_color')
    
    if not os.path.exists(test_dir):
        print("Test directory not found!")
        return
    
    test_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:test_images]
    
    plt.figure(figsize=(15, test_images * 3))
    
    for i, filename in enumerate(test_files):
        # Load and preprocess image
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_input = gray.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
        
        # Predict
        colorized = model.predict(gray_input, verbose=0)
        colorized = (colorized[0] * 255).astype(np.uint8)
        
        # Display results
        plt.subplot(test_images, 3, i * 3 + 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale Input')
        plt.axis('off')
        
        plt.subplot(test_images, 3, i * 3 + 2)
        plt.imshow(colorized)
        plt.title('AI Colorized')
        plt.axis('off')
        
        plt.subplot(test_images, 3, i * 3 + 3)
        plt.imshow(img)
        plt.title('Original Color')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Train the model
    model, history = train_model()
    
    # Test the model
    print("Testing model...")
    test_model(model)
    
    print("Training completed!")