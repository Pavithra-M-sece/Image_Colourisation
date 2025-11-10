# High Accuracy Training Configuration

# Model Parameters
IMG_SIZE = 128  # Image size (128x128 for faster training, 256x256 for better quality)
BATCH_SIZE = 16  # Reduce if you have memory issues
EPOCHS = 50  # Increase for better accuracy
LEARNING_RATE = 1e-3

# Data Parameters
MAX_TRAINING_IMAGES = 4000  # Increase for better accuracy (up to available data)
VALIDATION_SPLIT = 0.2
DATA_AUGMENTATION = True

# Training Parameters
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 8
MIN_LEARNING_RATE = 1e-7

# Paths
TRAIN_COLOR_PATH = r'archive (6)\data\train_color'
MODEL_SAVE_PATH = 'high_accuracy_colorization_model.keras'
FINAL_MODEL_PATH = 'final_high_accuracy_model.keras'

# Target Accuracy
TARGET_ACCURACY = 75.0  # Minimum accuracy percentage to achieve