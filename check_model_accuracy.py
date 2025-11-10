import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_test_data():
    """Load test data for accuracy evaluation"""
    color_folder = r'archive (6)\data\train_color'
    images = []
    count = 0
    
    for filename in os.listdir(color_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and count < 500:
            img_path = os.path.join(color_folder, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (128, 128))
                    images.append(img)
                    count += 1
            except:
                continue
    
    images = np.array(images)
    
    # Create grayscale versions
    gray_images = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray.reshape(128, 128, 1)
        gray_images.append(gray)
    
    gray_images = np.array(gray_images).astype('float32') / 255.0
    color_images = images.astype('float32') / 255.0
    
    return gray_images, color_images

def calculate_accuracy(model, X_test, y_test):
    """Calculate model accuracy"""
    predictions = model.predict(X_test, verbose=0)
    
    # Calculate MSE
    mse = np.mean((predictions - y_test) ** 2)
    
    # Calculate PSNR
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Convert PSNR to accuracy percentage
    if psnr >= 30:
        accuracy = 95 + min(5, (psnr - 30) * 0.5)
    elif psnr >= 25:
        accuracy = 85 + (psnr - 25) * 2
    elif psnr >= 20:
        accuracy = 70 + (psnr - 20) * 3
    else:
        accuracy = max(0, psnr * 3.5)
    
    return accuracy, psnr, mse

def main():
    """Check accuracy of existing model"""
    print("🔍 Checking Model Accuracy...")
    
    # Try to load existing model
    model_paths = [
        'best_colorization_model.keras',
        'final_colorization_model.keras',
        'auto_trained_model.keras',
        'colorization_model.keras'
    ]
    
    model = None
    model_name = None
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path, compile=False)
                model_name = path
                print(f"✅ Loaded model: {path}")
                break
            except:
                continue
    
    if model is None:
        print("❌ No trained model found!")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith('.keras'):
                print(f"  - {file}")
        return
    
    # Load test data
    print("📊 Loading test data...")
    try:
        X_test, y_test = load_test_data()
        print(f"Test samples: {len(X_test)}")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Calculate accuracy
    print("🎯 Calculating accuracy...")
    accuracy, psnr, mse = calculate_accuracy(model, X_test, y_test)
    
    # Display results
    print("\n" + "="*50)
    print("📈 MODEL ACCURACY RESULTS")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print(f"📊 PSNR: {psnr:.2f} dB")
    print(f"📉 MSE: {mse:.6f}")
    
    # Quality rating
    if accuracy >= 85:
        rating = "🏆 EXCELLENT"
    elif accuracy >= 75:
        rating = "🥇 VERY GOOD"
    elif accuracy >= 65:
        rating = "🥈 GOOD"
    elif accuracy >= 50:
        rating = "🥉 FAIR"
    else:
        rating = "❌ NEEDS IMPROVEMENT"
    
    print(f"Quality: {rating}")
    print("="*50)

if __name__ == "__main__":
    main()