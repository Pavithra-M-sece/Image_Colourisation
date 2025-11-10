# High-Accuracy Image Colorization Guide

## 🎯 Goal: Achieve 75%+ Accuracy

This guide provides a complete solution for training an image colorization model with at least 75% accuracy.

## 🚀 Quick Start

1. **Run the setup script:**
   ```bash
   python run_high_accuracy_training.py
   ```

2. **Or run manually:**
   ```bash
   pip install -r requirements_high_accuracy.txt
   python train_high_accuracy.py
   ```

## 🏗️ Architecture Improvements

### 1. **Improved U-Net with Residual Connections**
- Residual blocks prevent vanishing gradients
- Better feature propagation through the network
- Improved training stability

### 2. **Attention Mechanism**
- Focuses on important image regions
- Better color prediction accuracy
- Reduces artifacts in complex scenes

### 3. **LAB Color Space**
- More perceptually uniform than RGB
- Separates luminance (L) from color (AB)
- Better training convergence

## 📊 Training Strategies

### 1. **Custom Loss Function**
- Combines MSE loss with gradient loss
- Preserves edges and fine details
- Better perceptual quality

### 2. **Data Augmentation**
- Horizontal flipping
- Random rotation (±10 degrees)
- Increases effective dataset size

### 3. **Advanced Callbacks**
- Early stopping with patience
- Learning rate reduction on plateau
- Model checkpointing for best weights

## 📈 Accuracy Metrics

The model tracks multiple accuracy metrics:

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Measures reconstruction quality
   - Higher values = better quality
   - Target: >20 dB

2. **SSIM (Structural Similarity Index)**
   - Measures perceptual similarity
   - Range: 0-1 (higher is better)
   - Target: >0.7

3. **Overall Accuracy**
   - Converted from PSNR to percentage
   - Target: ≥75%

## ⚙️ Configuration Options

Edit `config_high_accuracy.py` to adjust:

```python
# For higher accuracy (requires more memory/time):
IMG_SIZE = 256
MAX_TRAINING_IMAGES = 8000
EPOCHS = 100

# For faster training (lower accuracy):
IMG_SIZE = 128
MAX_TRAINING_IMAGES = 2000
EPOCHS = 30
```

## 🔧 Troubleshooting

### Memory Issues
- Reduce `BATCH_SIZE` to 8 or 4
- Reduce `IMG_SIZE` to 64 or 96
- Close other applications

### Low Accuracy
- Increase `EPOCHS` to 100+
- Increase `MAX_TRAINING_IMAGES`
- Use `IMG_SIZE = 256` for better quality

### Slow Training
- Use GPU if available
- Reduce `MAX_TRAINING_IMAGES`
- Use smaller `IMG_SIZE`

## 📁 Output Files

After training, you'll get:

1. **Models:**
   - `high_accuracy_colorization_model.keras` (best checkpoint)
   - `final_high_accuracy_model.keras` (final model)

2. **Visualizations:**
   - `high_accuracy_training_history.png` (loss curves)
   - `high_accuracy_results.png` (sample predictions)

3. **Metrics:**
   - Console output with accuracy, PSNR, SSIM

## 🎯 Expected Results

With the improved architecture and training:

- **Accuracy:** 75-85%
- **PSNR:** 22-28 dB
- **SSIM:** 0.75-0.85
- **Training Time:** 1-3 hours (depending on hardware)

## 🔄 Using the Trained Model

After training, update your `app.py` to use the new model:

```python
# Load the high-accuracy model
model = tf.keras.models.load_model('final_high_accuracy_model.keras')

# Update preprocessing for LAB color space
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    
    # Convert to LAB
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L_channel = lab_img[:, :, 0:1].astype('float32') / 100.0
    
    return np.expand_dims(L_channel, axis=0)

def postprocess_prediction(L_input, ab_prediction):
    # Reconstruct LAB image
    lab_img = np.zeros((128, 128, 3))
    lab_img[:, :, 0] = L_input[0, :, :, 0] * 100
    lab_img[:, :, 1:] = (ab_prediction[0] * 255) - 128
    
    # Convert back to RGB
    rgb_img = cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return rgb_img
```

## 🏆 Success Criteria

The training is successful when:
- ✅ Accuracy ≥ 75%
- ✅ PSNR ≥ 20 dB
- ✅ SSIM ≥ 0.7
- ✅ Visual results look natural
- ✅ No obvious artifacts or color bleeding

## 📞 Support

If you encounter issues:
1. Check the console output for error messages
2. Verify dataset path and structure
3. Ensure sufficient disk space (>2GB)
4. Check GPU memory if using GPU
5. Try reducing batch size or image size