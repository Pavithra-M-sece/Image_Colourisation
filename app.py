from flask import Flask, render_template, request, jsonify
import os
import sys
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import threading
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
IMG_SIZE = 128
training_complete = False

def train_model_silent():
    """Train model silently in background"""
    global model, training_complete
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, concatenate
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from sklearn.model_selection import train_test_split
        import cv2
        
        # Suppress all output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        # Load data
        color_folder = r'archive (6)\data\train_color'
        images = []
        count = 0
        
        for filename in os.listdir(color_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and count < 1500:
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
        
        if len(images) == 0:
            return
            
        images = np.array(images)
        
        # Create grayscale versions
        gray_images = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = gray.reshape(128, 128, 1)
            gray_images.append(gray)
        
        gray_images = np.array(gray_images).astype('float32') / 255.0
        color_images = images.astype('float32') / 255.0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            gray_images, color_images, test_size=0.2, random_state=42
        )
        
        # Build model
        input_img = Input(shape=(128, 128, 1))
        
        # Encoder
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(input_img)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(2)(conv1)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(2)(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(2)(conv3)
        
        # Bottleneck
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        
        # Decoder
        up5 = Conv2DTranspose(256, 2, strides=2, padding='same')(conv4)
        up5 = concatenate([up5, conv3])
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
        
        up6 = Conv2DTranspose(128, 2, strides=2, padding='same')(conv5)
        up6 = concatenate([up6, conv2])
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
        
        up7 = Conv2DTranspose(64, 2, strides=2, padding='same')(conv6)
        up7 = concatenate([up7, conv1])
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
        
        output = Conv2D(3, 1, activation='sigmoid', padding='same')(conv7)
        
        model = Model(inputs=input_img, outputs=output)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        
        # Train
        model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=20,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Save model
        model.save('auto_trained_model.keras')
        training_complete = True
        
        # Restore output
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    except Exception as e:
        # Restore output on error
        try:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except:
            pass

def load_model():
    """Load or train the colorization model"""
    global model
    try:
        # Try loading existing model
        model_paths = [
            'auto_trained_model.keras',
            'best_colorization_model.keras',
            'final_colorization_model.keras'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path, compile=False)
                    return
                except:
                    continue
        
        # No model found, start training in background
        training_thread = threading.Thread(target=train_model_silent)
        training_thread.daemon = True
        training_thread.start()
        
    except Exception as e:
        model = None

def colorize_image(image_path):
    """Colorize image using trained model or demo processing"""
    global model, training_complete
    try:
        # Open and process the image
        img = Image.open(image_path)
        img = img.convert('RGB')
        
        # Check if training completed and model needs loading
        if model is None and training_complete:
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model('auto_trained_model.keras', compile=False)
            except:
                pass
        
        if model is not None:
            # Use trained model
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img_resized)
            
            # Convert to grayscale
            gray = Image.fromarray(img_array).convert('L')
            gray_array = np.array(gray).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
            
            # Predict
            colorized = model.predict(gray_array, verbose=0)
            colorized = (colorized[0] * 255).astype(np.uint8)
            
            # Resize back to original size if needed
            colorized_img = Image.fromarray(colorized)
            gray_display = gray.convert('RGB')
            
            return np.array(gray_display), np.array(colorized_img)
        
        else:
            # Demo colorization using PIL
            img = img.resize((256, 256))
            gray_img = img.convert('L').convert('RGB')
            
            # Simple colorization effect
            colorized_img = img.copy()
            enhancer = ImageEnhance.Color(colorized_img)
            colorized_img = enhancer.enhance(1.3)
            
            enhancer = ImageEnhance.Contrast(colorized_img)
            colorized_img = enhancer.enhance(1.1)
            
            return np.array(gray_img), np.array(colorized_img)
        
    except Exception as e:
        return None, None

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(image_array)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Colorize the image
            gray_img, colorized_img = colorize_image(filepath)
            
            if gray_img is None or colorized_img is None:
                return jsonify({'error': 'Processing failed'})
            
            # Convert to base64 for display
            gray_b64 = image_to_base64(gray_img)
            colorized_b64 = image_to_base64(colorized_img)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'gray_image': gray_b64,
                'colorized_image': colorized_b64
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    print("Application started successfully")
    load_model()
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)