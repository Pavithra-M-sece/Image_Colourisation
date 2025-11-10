# Improved Model Training Guide

## Training Options

### 1. Quick Training (Recommended for Testing)
```bash
python quick_train.py
```
- **Time**: ~30-60 minutes
- **Model**: Simple U-Net
- **Images**: 1000 samples
- **Resolution**: 128x128
- **Good for**: Testing and quick results

### 2. Full Training (Best Quality)
```bash
python train_improved.py
```
- **Time**: 4-8 hours
- **Model**: Advanced U-Net with attention
- **Images**: All available
- **Resolution**: 256x256
- **Good for**: Production-quality results

## Model Improvements

### Architecture Enhancements
- **Attention Gates**: Focus on relevant features
- **Residual Connections**: Better gradient flow
- **Squeeze-Excitation**: Channel attention
- **Batch Normalization**: Stable training
- **Dropout**: Prevent overfitting

### Training Improvements
- **Data Augmentation**: Horizontal flip, brightness, contrast, noise
- **Advanced Loss Functions**: MSE + SSIM + Perceptual loss
- **Learning Rate Scheduling**: Adaptive learning rate
- **Early Stopping**: Prevent overfitting
- **Model Checkpointing**: Save best model
- **Mixed Precision**: Faster training on modern GPUs

### Data Processing
- **Efficient Data Loading**: Custom generator for memory efficiency
- **Image Preprocessing**: Proper normalization and resizing
- **Validation Split**: 20% for model evaluation
- **Augmentation Pipeline**: Realistic image variations

## Configuration

Edit `train_config.py` to customize:

```python
# Model settings
IMG_SIZE = 256          # Higher = better quality, slower training
BATCH_SIZE = 8          # Lower if you have memory issues
EPOCHS = 50             # More epochs = better training
LEARNING_RATE = 1e-4    # Lower = more stable training

# Architecture options
USE_ATTENTION = True     # Enable attention mechanisms
USE_RESIDUAL_BLOCKS = True  # Enable residual connections
DROPOUT_RATE = 0.5      # Prevent overfitting

# Loss function weights
MSE_WEIGHT = 0.6        # Pixel-wise accuracy
SSIM_WEIGHT = 0.3       # Structural similarity
PERCEPTUAL_WEIGHT = 0.1 # Perceptual quality
```

## Hardware Requirements

### Minimum (CPU Training)
- **RAM**: 8GB
- **Storage**: 5GB free space
- **Time**: 2-4 hours for quick training

### Recommended (GPU Training)
- **GPU**: NVIDIA GTX 1060 or better
- **VRAM**: 6GB+
- **RAM**: 16GB
- **Time**: 30-60 minutes for quick training

### Optimal (High-end GPU)
- **GPU**: RTX 3070 or better
- **VRAM**: 8GB+
- **RAM**: 32GB
- **Time**: 15-30 minutes for quick training

## Training Tips

### For Better Results
1. **Use more data**: Increase `MAX_IMAGES` or set to `None`
2. **Higher resolution**: Set `IMG_SIZE = 256` or `512`
3. **More epochs**: Increase `EPOCHS` to 50-100
4. **Data augmentation**: Enable all augmentation options
5. **Advanced loss**: Use combined loss with perceptual component

### For Faster Training
1. **Reduce image size**: Set `IMG_SIZE = 64` or `128`
2. **Smaller batch**: Reduce `BATCH_SIZE` to 4-8
3. **Fewer images**: Set `MAX_IMAGES = 500-1000`
4. **Simple loss**: Use only MSE loss
5. **CPU training**: Disable mixed precision

### Memory Issues
1. **Reduce batch size**: Set `BATCH_SIZE = 4` or `2`
2. **Smaller images**: Set `IMG_SIZE = 64`
3. **Limit data**: Set `MAX_IMAGES = 500`
4. **Disable augmentation**: Set `augment=False`

## Monitoring Training

### TensorBoard (Advanced Training)
```bash
tensorboard --logdir=logs/fit
```
Open http://localhost:6006 to view training progress

### Training Plots
- **Loss curves**: Monitor overfitting
- **PSNR/SSIM**: Quality metrics
- **Sample predictions**: Visual progress

## Model Files

After training, you'll have:
- `colorization_model.keras` - Main model for web app
- `best_colorization_model.keras` - Best checkpoint
- `final_colorization_model.keras` - Final trained model
- `training_history.png` - Training plots
- `test_results.png` - Sample results

## Troubleshooting

### Common Issues
1. **Out of memory**: Reduce batch size and image size
2. **Slow training**: Use GPU, reduce data size
3. **Poor results**: Train longer, use more data
4. **Model not loading**: Check file paths and permissions

### Performance Optimization
1. **Enable GPU**: Install tensorflow-gpu
2. **Mixed precision**: Enable in config for RTX cards
3. **Data pipeline**: Use tf.data for better performance
4. **Parallel processing**: Enable in data loader