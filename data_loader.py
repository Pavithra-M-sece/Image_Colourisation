import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import albumentations as A
from train_config import *

class ColorDataGenerator(Sequence):
    """Custom data generator for efficient loading and augmentation"""
    
    def __init__(self, image_paths, batch_size=BATCH_SIZE, img_size=IMG_SIZE, 
                 augment=True, shuffle=True):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        # Data augmentation pipeline
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.2),
            A.RandomRotate90(p=0.2),
        ]) if augment else None
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indices]
        
        X, y = self._generate_batch(batch_paths)
        return X, y
    
    def _generate_batch(self, batch_paths):
        X = np.zeros((self.batch_size, self.img_size, self.img_size, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        
        for i, path in enumerate(batch_paths):
            try:
                # Load image
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
                
                # Apply augmentation
                if self.augmentation is not None:
                    augmented = self.augmentation(image=img)
                    img = augmented['image']
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                # Normalize
                X[i] = gray.reshape(self.img_size, self.img_size, 1) / 255.0
                y[i] = img / 255.0
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Fill with zeros if error occurs
                continue
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_data_generators(data_dir, validation_split=0.2, max_images=None):
    """Create training and validation data generators"""
    
    color_dir = os.path.join(data_dir, 'train_color')
    
    if not os.path.exists(color_dir):
        raise ValueError(f"Directory {color_dir} not found!")
    
    # Get all image paths
    image_files = [f for f in os.listdir(color_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    if max_images:
        image_files = image_files[:max_images]
    
    image_paths = [os.path.join(color_dir, f) for f in image_files]
    
    # Split into train and validation
    split_idx = int(len(image_paths) * (1 - validation_split))
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    
    # Create generators
    train_gen = ColorDataGenerator(
        train_paths, 
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        augment=True,
        shuffle=True
    )
    
    val_gen = ColorDataGenerator(
        val_paths,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        augment=False,
        shuffle=False
    )
    
    return train_gen, val_gen

def load_test_images(data_dir, num_images=5):
    """Load test images for evaluation"""
    test_dir = os.path.join(data_dir, 'test_color')
    
    if not os.path.exists(test_dir):
        # Fallback to train_color if test_color doesn't exist
        test_dir = os.path.join(data_dir, 'train_color')
    
    test_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))][:num_images]
    
    images = []
    grays = []
    
    for filename in test_files:
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        images.append(img)
        grays.append(gray)
    
    return images, grays