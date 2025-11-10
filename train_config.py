"""
Configuration file for training parameters
Modify these settings based on your hardware and requirements
"""

# Model Configuration
IMG_SIZE = 256  # Image resolution (128, 256, 512)
BATCH_SIZE = 8  # Reduce if you have memory issues
EPOCHS = 50
LEARNING_RATE = 1e-4

# Data Configuration
MAX_IMAGES = None  # Set to None to use all images, or specify a number for testing
VALIDATION_SPLIT = 0.2
AUGMENTATION_PROBABILITY = 0.7

# Training Configuration
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

# Loss Function Weights
MSE_WEIGHT = 0.6
SSIM_WEIGHT = 0.3
PERCEPTUAL_WEIGHT = 0.1

# Model Architecture
USE_ATTENTION = True
USE_RESIDUAL_BLOCKS = True
DROPOUT_RATE = 0.5
FILTERS_BASE = 64  # Base number of filters (will be multiplied by 2 at each level)

# Paths
DATA_DIR = './archive (6)/data'
MODEL_SAVE_PATH = 'best_colorization_model.keras'
FINAL_MODEL_PATH = 'final_colorization_model.keras'
HISTORY_PLOT_PATH = 'training_history.png'
TEST_RESULTS_PATH = 'test_results.png'

# Hardware Configuration
USE_MIXED_PRECISION = True  # Enable for faster training on modern GPUs
PREFETCH_BUFFER = tf.data.AUTOTUNE if 'tf' in globals() else 2