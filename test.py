import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for testing"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    
    # Create grayscale version
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray.reshape(128, 128, 1)
    
    # Normalize
    gray = gray.astype('float32') / 255.0
    img = img.astype('float32') / 255.0
    
    return gray, img

def test_model(model_path, test_image_path):
    """Test the trained model on a single image"""
    # Load model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Load and preprocess test image
    gray_img, original_img = load_and_preprocess_image(test_image_path)
    
    # Add batch dimension
    gray_batch = np.expand_dims(gray_img, axis=0)
    
    # Predict
    prediction = model.predict(gray_batch)
    predicted_img = prediction[0]
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(gray_img.squeeze(), cmap='gray')
    plt.title('Input Grayscale')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_img)
    plt.title('AI Colorized')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(original_img)
    plt.title('Original Color')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return predicted_img

def test_multiple_images(model_path, test_folder, num_images=5):
    """Test model on multiple images"""
    model = load_model(model_path)
    
    # Get test images
    test_files = [f for f in os.listdir(test_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_images]
    
    fig, axes = plt.subplots(3, num_images, figsize=(15, 9))
    
    for i, filename in enumerate(test_files):
        img_path = os.path.join(test_folder, filename)
        gray_img, original_img = load_and_preprocess_image(img_path)
        
        # Predict
        gray_batch = np.expand_dims(gray_img, axis=0)
        prediction = model.predict(gray_batch, verbose=0)
        predicted_img = prediction[0]
        
        # Plot
        axes[0, i].imshow(gray_img.squeeze(), cmap='gray')
        axes[0, i].set_title('Grayscale')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(predicted_img)
        axes[1, i].set_title('AI Colorized')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(original_img)
        axes[2, i].set_title('Original')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('multiple_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Test single image
    model_path = 'colorization_model.keras'
    test_image = r'archive (6)\data\test_color\test_image.jpg'  # Replace with actual test image
    
    if os.path.exists(model_path):
        if os.path.exists(test_image):
            test_model(model_path, test_image)
        else:
            print(f"Test image not found: {test_image}")
            # Test on multiple images from folder
            test_folder = r'archive (6)\data\test_color'
            if os.path.exists(test_folder):
                test_multiple_images(model_path, test_folder)
            else:
                print("No test folder found")
    else:
        print(f"Model not found: {model_path}")
        print("Please train the model first using train_model.py")