import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path, max_images=1000):
    """Load images from folder"""
    images = []
    count = 0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            if max_images and count >= max_images:
                break
                
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (128, 128))
                    images.append(img)
                    count += 1
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    return np.array(images)

def prepare_data(color_folder, max_images=1000):
    """Load and prepare training data"""
    # Load color images
    color_images = load_images_from_folder(color_folder, max_images)
    
    if len(color_images) == 0:
        raise ValueError("No images found!")
    
    # Create grayscale versions
    gray_images = []
    for img in color_images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray.reshape(128, 128, 1)
        gray_images.append(gray)
    
    gray_images = np.array(gray_images)
    
    # Normalize to [0, 1]
    gray_images = gray_images.astype('float32') / 255.0
    color_images = color_images.astype('float32') / 255.0
    
    return gray_images, color_images

def split_data(X_data, y_data, test_size=0.2):
    """Split data into train and test sets"""
    return train_test_split(X_data, y_data, test_size=test_size, random_state=42)

if __name__ == "__main__":
    color_folder = r'archive (6)\data\train_color'
    X_data, y_data = prepare_data(color_folder, max_images=1000)
    X_train, X_test, y_train, y_test = split_data(X_data, y_data)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Gray images shape: {X_train.shape}")
    print(f"Color images shape: {y_train.shape}")