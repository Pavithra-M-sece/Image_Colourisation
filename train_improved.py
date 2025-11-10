"""
Improved training script for image colorization
Run this to train a better model with advanced techniques
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from datetime import datetime

# Import custom modules
from train_config import *
from model_architecture import build_unet_model
from loss_functions import get_loss_function, psnr_metric, ssim_metric
from data_loader import create_data_generators, load_test_images

def setup_training():
    """Setup training environment"""
    
    # Enable mixed precision for faster training
    if USE_MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
    
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, using CPU")

def create_callbacks():
    """Create training callbacks"""
    
    # Create logs directory
    log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    return callbacks

def train_model():
    """Main training function"""
    
    print("Setting up training environment...")
    setup_training()
    
    print("Creating data generators...")
    try:
        train_gen, val_gen = create_data_generators(
            DATA_DIR, 
            validation_split=VALIDATION_SPLIT,
            max_images=MAX_IMAGES
        )
    except Exception as e:
        print(f"Error creating data generators: {e}")
        return None, None
    
    print("Building model...")
    model = build_unet_model()
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss_fn = get_loss_function('combined')
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['mae', 'mse', psnr_metric, ssim_metric]
    )
    
    print("Model architecture:")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    print("Starting training...")
    print(f"Training samples: {len(train_gen) * BATCH_SIZE}")
    print(f"Validation samples: {len(val_gen) * BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH}")
    
    return model, history

def plot_training_history(history):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE
    axes[0, 1].plot(history.history['mae'], label='Training MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # PSNR
    if 'psnr_metric' in history.history:
        axes[1, 0].plot(history.history['psnr_metric'], label='Training PSNR')
        axes[1, 0].plot(history.history['val_psnr_metric'], label='Validation PSNR')
        axes[1, 0].set_title('Peak Signal-to-Noise Ratio')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PSNR')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # SSIM
    if 'ssim_metric' in history.history:
        axes[1, 1].plot(history.history['ssim_metric'], label='Training SSIM')
        axes[1, 1].plot(history.history['val_ssim_metric'], label='Validation SSIM')
        axes[1, 1].set_title('Structural Similarity Index')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model):
    """Evaluate the trained model"""
    
    print("Loading test images...")
    try:
        test_images, test_grays = load_test_images(DATA_DIR, num_images=8)
    except Exception as e:
        print(f"Error loading test images: {e}")
        return
    
    print("Generating predictions...")
    
    fig, axes = plt.subplots(len(test_images), 3, figsize=(12, len(test_images) * 3))
    
    for i, (original, gray) in enumerate(zip(test_images, test_grays)):
        # Prepare input
        gray_input = gray.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
        
        # Predict
        colorized = model.predict(gray_input, verbose=0)
        colorized = (colorized[0] * 255).astype(np.uint8)
        
        # Plot results
        if len(test_images) == 1:
            axes[0].imshow(gray, cmap='gray')
            axes[0].set_title('Grayscale Input')
            axes[0].axis('off')
            
            axes[1].imshow(colorized)
            axes[1].set_title('AI Colorized')
            axes[1].axis('off')
            
            axes[2].imshow(original)
            axes[2].set_title('Original Color')
            axes[2].axis('off')
        else:
            axes[i, 0].imshow(gray, cmap='gray')
            axes[i, 0].set_title('Grayscale Input')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(colorized)
            axes[i, 1].set_title('AI Colorized')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(original)
            axes[i, 2].set_title('Original Color')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(TEST_RESULTS_PATH, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline"""
    
    print("=" * 50)
    print("IMPROVED IMAGE COLORIZATION TRAINING")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found!")
        print("Please ensure your dataset is in the correct location.")
        return
    
    # Train model
    model, history = train_model()
    
    if model is None:
        print("Training failed!")
        return
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model)
    
    print("Training completed successfully!")
    print(f"Best model saved as: {MODEL_SAVE_PATH}")
    print(f"Final model saved as: {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()