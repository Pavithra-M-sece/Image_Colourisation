# AI Image Colorization Project

This project uses deep learning to automatically colorize black and white images. It features a trained CNN model and a modern web interface built with Flask.

## Features

- **Deep Learning Model**: Uses a U-Net style CNN architecture for image colorization
- **Web Interface**: Modern, responsive web application with drag-and-drop functionality
- **Real-time Processing**: Fast image processing and colorization
- **Download Results**: Download colorized images directly from the web interface

## Project Structure

```
Image_colourisation_AIML/
├── app.py                 # Main Flask application
├── model_converter.py     # Script to convert/save the trained model
├── requirements.txt       # Python dependencies
├── Untitled11.ipynb     # Original Colab notebook with model training
├── templates/
│   └── index.html        # Main web interface template
├── static/
│   ├── css/
│   │   └── style.css     # Styling for the web interface
│   └── js/
│       └── script.js     # JavaScript for interactivity
├── uploads/              # Temporary folder for uploaded images
└── archive (6)/          # Dataset folder
    └── data/
        ├── train_color/  # Training color images
        ├── train_black/  # Training grayscale images
        ├── test_color/   # Test color images
        └── test_black/   # Test grayscale images
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Model

If you have a trained model from the Colab notebook:

```bash
python model_converter.py
```

This will either:
- Convert your existing trained model from Google Drive
- Create a new model with the same architecture if no trained model is found

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Model Architecture

The colorization model uses a U-Net style architecture:

- **Input**: Grayscale images (128x128x1)
- **Output**: RGB colored images (128x128x3)
- **Architecture**: Encoder-decoder with skip connections
- **Training**: Trained on paired grayscale and color images

### Model Details

- **Encoder**: Downsampling layers with Conv2D and BatchNormalization
- **Bottleneck**: Feature extraction layers
- **Decoder**: Upsampling layers with skip connections from encoder
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 1e-3

## Usage

1. **Open the Web Interface**: Navigate to `http://localhost:5000`
2. **Upload an Image**: 
   - Drag and drop an image onto the upload area, or
   - Click "Choose File" to browse and select an image
3. **Wait for Processing**: The AI model will process your image
4. **View Results**: See the original grayscale and colorized versions side by side
5. **Download**: Click "Download Colorized Image" to save the result

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)

## Technical Details

### Web Framework
- **Flask**: Lightweight Python web framework
- **HTML5**: Modern semantic markup
- **CSS3**: Responsive design with gradients and animations
- **JavaScript**: Interactive functionality and AJAX requests

### Image Processing
- **OpenCV**: Image loading and preprocessing
- **TensorFlow**: Deep learning model inference
- **PIL/Pillow**: Image format conversion
- **NumPy**: Numerical operations

### Model Training (from Colab notebook)
- **Dataset**: Image colorization dataset from Kaggle
- **Training Images**: ~4000 color images converted to grayscale pairs
- **Epochs**: 20 epochs with early stopping
- **Batch Size**: 16
- **Validation Split**: 20%

## Performance

- **Processing Time**: ~2-3 seconds per image
- **Model Size**: ~17.6 MB
- **Input Resolution**: 128x128 pixels
- **Memory Usage**: ~500MB during inference

## Customization

### Modify Model Architecture
Edit the `build_colorization_model()` function in `app.py` or `model_converter.py`

### Change Image Size
Update the `IMG_SIZE` variable (note: requires retraining)

### Styling
Modify `static/css/style.css` for custom styling

### Add Features
Extend `app.py` with additional routes and functionality

## Troubleshooting

### Model Not Loading
- Ensure the model file exists: `colorization_model.keras`
- Run `python model_converter.py` to create/convert the model

### Memory Issues
- Reduce batch size or image resolution
- Ensure sufficient RAM (minimum 4GB recommended)

### Slow Processing
- Use GPU acceleration if available
- Optimize model for inference (quantization, pruning)

## Future Improvements

- [ ] Support for higher resolution images
- [ ] Batch processing multiple images
- [ ] Different colorization styles/models
- [ ] Mobile app version
- [ ] API endpoints for integration
- [ ] User feedback and rating system

## License

This project is for educational purposes. Please ensure you have the right to use any images you process.

## Acknowledgments

- Dataset from Kaggle: Image Colorization Dataset
- TensorFlow and Keras for deep learning framework
- Flask for web framework
- OpenCV for image processing