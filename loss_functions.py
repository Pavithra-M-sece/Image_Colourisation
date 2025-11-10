import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from train_config import *

class PerceptualLoss:
    """Perceptual loss using VGG19 features"""
    
    def __init__(self, layer_names=['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']):
        # Load VGG19
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
        vgg.trainable = False
        
        # Extract features from specified layers
        outputs = [vgg.get_layer(name).output for name in layer_names]
        self.feature_extractor = Model(vgg.input, outputs)
        
    def __call__(self, y_true, y_pred):
        # Extract features
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)
        
        # Calculate loss
        loss = 0
        for true_feat, pred_feat in zip(true_features, pred_features):
            loss += tf.reduce_mean(tf.square(true_feat - pred_feat))
        
        return loss

def ssim_loss(y_true, y_pred):
    """SSIM-based loss function"""
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def l1_loss(y_true, y_pred):
    """L1 (MAE) loss"""
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def l2_loss(y_true, y_pred):
    """L2 (MSE) loss"""
    return tf.reduce_mean(tf.square(y_true - y_pred))

def combined_loss(y_true, y_pred):
    """Combined loss function with multiple components"""
    
    # Basic losses
    mse = l2_loss(y_true, y_pred)
    mae = l1_loss(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    
    # Combine losses
    total_loss = (
        MSE_WEIGHT * mse +
        0.2 * mae +  # Small MAE component for sharpness
        SSIM_WEIGHT * ssim
    )
    
    return total_loss

def adversarial_loss(y_true, y_pred):
    """Adversarial loss for GAN training"""
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(y_pred), logits=y_pred
    ))

def discriminator_loss(real_output, fake_output):
    """Discriminator loss for GAN training"""
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(real_output), logits=real_output
    ))
    
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(fake_output), logits=fake_output
    ))
    
    return real_loss + fake_loss

def color_histogram_loss(y_true, y_pred):
    """Loss based on color histogram similarity"""
    def compute_histogram(image):
        # Convert to HSV for better color representation
        hsv = tf.image.rgb_to_hsv(image)
        
        # Compute histograms for each channel
        h_hist = tf.histogram_fixed_width(hsv[..., 0], [0.0, 1.0], nbins=64)
        s_hist = tf.histogram_fixed_width(hsv[..., 1], [0.0, 1.0], nbins=64)
        v_hist = tf.histogram_fixed_width(hsv[..., 2], [0.0, 1.0], nbins=64)
        
        return tf.concat([h_hist, s_hist, v_hist], axis=0)
    
    true_hist = compute_histogram(y_true)
    pred_hist = compute_histogram(y_pred)
    
    # Normalize histograms
    true_hist = tf.cast(true_hist, tf.float32)
    pred_hist = tf.cast(pred_hist, tf.float32)
    
    true_hist = true_hist / tf.reduce_sum(true_hist)
    pred_hist = pred_hist / tf.reduce_sum(pred_hist)
    
    # Calculate histogram intersection
    intersection = tf.reduce_sum(tf.minimum(true_hist, pred_hist))
    
    return 1.0 - intersection

def get_loss_function(loss_type='combined'):
    """Get the specified loss function"""
    if loss_type == 'mse':
        return l2_loss
    elif loss_type == 'mae':
        return l1_loss
    elif loss_type == 'ssim':
        return ssim_loss
    elif loss_type == 'combined':
        return combined_loss
    elif loss_type == 'perceptual':
        return PerceptualLoss()
    elif loss_type == 'histogram':
        return color_histogram_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Custom metrics
def psnr_metric(y_true, y_pred):
    """Peak Signal-to-Noise Ratio"""
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    """Structural Similarity Index"""
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))