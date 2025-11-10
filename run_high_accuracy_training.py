"""
Quick setup and run script for high-accuracy image colorization training
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_high_accuracy.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def check_data_folder():
    """Check if data folder exists"""
    data_path = r'archive (6)\data\train_color'
    if os.path.exists(data_path):
        num_images = len([f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✅ Data folder found with {num_images} training images")
        return True
    else:
        print(f"❌ Data folder not found: {data_path}")
        print("Please ensure the dataset is extracted in the correct location")
        return False

def run_training():
    """Run the high-accuracy training script"""
    print("Starting high-accuracy training...")
    try:
        subprocess.check_call([sys.executable, "train_high_accuracy.py"])
        print("✅ Training completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False
    return True

def main():
    """Main setup and run function"""
    print("🚀 High-Accuracy Image Colorization Training Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        return
    
    # Step 2: Check data
    if not check_data_folder():
        return
    
    # Step 3: Run training
    print("\n🎯 Starting training to achieve 75%+ accuracy...")
    print("This may take 1-3 hours depending on your hardware.")
    print("The script will:")
    print("- Use improved U-Net architecture with attention")
    print("- Apply data augmentation")
    print("- Use LAB color space for better results")
    print("- Track accuracy metrics (PSNR, SSIM)")
    
    user_input = input("\nProceed with training? (y/n): ")
    if user_input.lower() == 'y':
        run_training()
    else:
        print("Training cancelled.")

if __name__ == "__main__":
    main()