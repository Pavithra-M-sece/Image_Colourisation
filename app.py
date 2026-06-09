from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import hashlib
import warnings
from datetime import datetime
from database import (init_db, create_user, verify_user, save_colorization,
                      get_user_colorizations, get_all_users, get_admin_stats,
                      get_recent_activity, toggle_user_status, delete_user, update_user_role)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['COLORIZED_FOLDER'] = 'colorized'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'change-this-secret-key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['COLORIZED_FOLDER'], exist_ok=True)

init_db()

model     = None
IMG_SIZE  = 128
MODEL_TYPE = None  # 'lab_unet', 'rgb_unet', or None

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    global model, MODEL_TYPE
    try:
        import tensorflow as tf
        # Priority 1: LAB U-Net from Colab (best quality)
        if os.path.exists('colorization_lab_unet.keras'):
            model = tf.keras.models.load_model('colorization_lab_unet.keras', compile=False)
            MODEL_TYPE = 'lab_unet'
            print("LAB U-Net loaded: colorization_lab_unet.keras")
            return
        # Priority 2: old RGB model fallback
        for path in ['best_colorization_model.keras', 'auto_trained_model.keras']:
            if os.path.exists(path):
                model = tf.keras.models.load_model(path, compile=False)
                MODEL_TYPE = 'rgb_unet'
                print(f"RGB model loaded (fallback): {path}")
                return
    except Exception as e:
        print(f"Model load error: {e}")
    print("No trained model. Train in Colab and download colorization_lab_unet.keras")

# ── Colorization ──────────────────────────────────────────────────────────────

def colorize_image(image_path):
    global model, MODEL_TYPE
    try:
        import cv2
        img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img.size
        gray_display = np.array(img.convert('L').convert('RGB'))

        # ── LAB U-Net (Colab trained — correct colors, sharp) ────────────────
        if model is not None and MODEL_TYPE == 'lab_unet':
            # Resize to model input size
            img_resized = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
            bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

            # Normalize L channel: 0-100 → 0-1
            L = lab[:, :, 0:1] / 100.0
            L_batch = np.expand_dims(L, axis=0)

            # Predict ab channels (output: -1 to 1)
            ab_pred = model.predict(L_batch, verbose=0)[0]

            # Get full-resolution L from original image for sharp output
            img_full = np.array(img.resize((orig_w, orig_h)))
            bgr_full = cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR)
            lab_full = cv2.cvtColor(bgr_full.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
            L_full = lab_full[:, :, 0]  # Full resolution, sharp

            # Upscale predicted ab to original resolution
            ab_denorm = ab_pred * 128.0
            ab_up = cv2.resize(ab_denorm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            # Reconstruct LAB at full resolution with sharp L
            lab_out = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
            lab_out[:, :, 0] = L_full
            lab_out[:, :, 1:] = ab_up

            bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
            rgb_out = cv2.cvtColor(
                np.clip(bgr_out * 255, 0, 255).astype(np.uint8),
                cv2.COLOR_BGR2RGB
            )
            colorized_pil = Image.fromarray(rgb_out)
            # Light enhancement — don't over-process
            colorized_pil = ImageEnhance.Color(colorized_pil).enhance(1.2)
            colorized_pil = ImageEnhance.Sharpness(colorized_pil).enhance(1.3)
            return gray_display, np.array(colorized_pil)

        # ── Old RGB U-Net (fallback, tends to produce gray) ───────────────────
        elif model is not None and MODEL_TYPE == 'rgb_unet':
            img_r = img.resize((IMG_SIZE, IMG_SIZE))
            gray_arr = np.array(img_r.convert('L')).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
            colorized = model.predict(gray_arr, verbose=0)
            colorized = (colorized[0] * 255).astype(np.uint8)
            colorized_out = np.array(Image.fromarray(colorized).resize((orig_w, orig_h)))
            return gray_display, colorized_out

        # ── No model: demo enhancement ────────────────────────────────────────
        else:
            img_r = img.resize((256, 256))
            colorized_img = ImageEnhance.Color(img_r.copy()).enhance(1.3)
            return np.array(img_r.convert('L').convert('RGB')), np.array(colorized_img)

    except Exception as e:
        print(f"Colorize error: {e}")
        return None, None

# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_base64(image_array):
    img = Image.fromarray(image_array)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS

def try_open_as_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        return Image.open(filepath).convert('RGB')
    except Exception:
        return None

def convert_to_png(filepath):
    img = Image.open(filepath).convert('RGB')
    new_path = os.path.splitext(filepath)[0] + '_converted.png'
    img.save(new_path, 'PNG')
    return new_path

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        user = verify_user(username, password)
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[2]
            return redirect(url_for('admin_dashboard') if user[2] == 'admin' else url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        email    = request.form.get('email')
        if create_user(username, password, email):
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        flash('Username already exists', 'error')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/clear')
def clear_session():
    session.clear()
    return redirect(url_for('login'))

# ── User routes ───────────────────────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    colorizations = get_user_colorizations(session['user_id'])
    return render_template('dashboard.html', username=session.get('username'), colorizations=colorizations)

@app.route('/colorize')
@login_required
def colorize_page():
    return render_template('index.html', username=session.get('username'))

@app.route('/history')
@login_required
def history():
    colorizations = get_user_colorizations(session['user_id'])
    return render_template('dashboard.html', username=session.get('username'), colorizations=colorizations)

@app.route('/check-file', methods=['POST'])
@login_required
def check_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file selected'})
    file = request.files['file']
    ext  = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''

    if allowed_file(file.filename):
        return jsonify({'status': 'ok'})

    filename = secure_filename(file.filename)
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tmp_' + filename)
    file.save(tmp_path)

    if try_open_as_image(tmp_path):
        return jsonify({'status': 'needs_conversion', 'ext': ext.upper() or 'UNKNOWN',
                        'filename': file.filename, 'tmp': 'tmp_' + filename})
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return jsonify({'status': 'unsupported', 'ext': ext.upper() or 'UNKNOWN'})

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    convert  = request.form.get('convert') == 'true'
    tmp_name = request.form.get('tmp')

    if tmp_name:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(tmp_name))
        filename = tmp_name
    else:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'error': 'No file selected'})
        file     = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Upload failed, please try again'})

    converted = False
    if not allowed_file(filename):
        if not convert:
            return jsonify({'error': 'Conversion not permitted by user'})
        try:
            filepath  = convert_to_png(filepath)
            filename  = os.path.basename(filepath)
            converted = True
        except Exception:
            return jsonify({'error': 'Conversion failed — file may be corrupted.'})

    gray_img, colorized_img = colorize_image(filepath)
    if gray_img is None:
        return jsonify({'error': 'Colorization failed'})

    colorized_filename = f"colorized_{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    colorized_path     = os.path.join(app.config['COLORIZED_FOLDER'], colorized_filename)
    Image.fromarray(colorized_img).save(colorized_path)
    save_colorization(session['user_id'], filename, filepath, colorized_path)

    return jsonify({'success': True,
                    'gray_image': image_to_base64(gray_img),
                    'colorized_image': image_to_base64(colorized_img),
                    'converted': converted})

# ── Admin routes ──────────────────────────────────────────────────────────────

@app.route('/admin')
@admin_required
def admin_dashboard():
    return render_template('admin.html', username=session.get('username'),
                           stats=get_admin_stats(), users=get_all_users(),
                           activity=get_recent_activity(10))

@app.route('/admin/user/toggle/<user_id>', methods=['POST'])
@admin_required
def admin_toggle_user(user_id):
    result = toggle_user_status(user_id)
    return jsonify({'success': result is not None, 'is_active': result})

@app.route('/admin/user/delete/<user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    return jsonify({'success': delete_user(user_id)})

@app.route('/admin/user/role/<user_id>', methods=['POST'])
@admin_required
def admin_update_role(user_id):
    role = request.json.get('role')
    if role in ('user', 'admin') and update_user_role(user_id, role):
        return jsonify({'success': True})
    return jsonify({'success': False})

if __name__ == '__main__':
    print("Starting AI Image Colorization...")
    load_model()
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
