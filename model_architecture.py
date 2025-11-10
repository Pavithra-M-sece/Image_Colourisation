import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from train_config import *

def conv_block(x, filters, kernel_size=3, strides=1, activation='relu'):
    """Standard convolutional block"""
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x

def residual_block(x, filters):
    """Residual block with skip connection"""
    shortcut = x
    
    x = conv_block(x, filters)
    x = conv_block(x, filters, activation=None)
    
    # Match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def attention_gate(g, x, filters):
    """Attention gate for focusing on relevant features"""
    # Gating signal
    g1 = Conv2D(filters, 1, padding='same')(g)
    g1 = BatchNormalization()(g1)
    
    # Feature map
    x1 = Conv2D(filters, 1, padding='same')(x)
    x1 = BatchNormalization()(x1)
    
    # Attention coefficients
    psi = Add()([g1, x1])
    psi = Activation('relu')(psi)
    psi = Conv2D(1, 1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    
    # Apply attention
    return Multiply()([x, psi])

def squeeze_excitation_block(x, ratio=16):
    """Squeeze-and-Excitation block for channel attention"""
    filters = x.shape[-1]
    
    # Squeeze
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, filters))(se)
    
    # Excitation
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    
    # Scale
    return Multiply()([x, se])

def build_unet_model():
    """Build improved U-Net model with attention and residual connections"""
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder
    e1 = conv_block(inputs, FILTERS_BASE)
    if USE_RESIDUAL_BLOCKS:
        e1 = residual_block(e1, FILTERS_BASE)
    e1 = squeeze_excitation_block(e1)
    p1 = MaxPooling2D(2)(e1)
    
    e2 = conv_block(p1, FILTERS_BASE * 2)
    if USE_RESIDUAL_BLOCKS:
        e2 = residual_block(e2, FILTERS_BASE * 2)
    e2 = squeeze_excitation_block(e2)
    p2 = MaxPooling2D(2)(e2)
    
    e3 = conv_block(p2, FILTERS_BASE * 4)
    if USE_RESIDUAL_BLOCKS:
        e3 = residual_block(e3, FILTERS_BASE * 4)
    e3 = squeeze_excitation_block(e3)
    p3 = MaxPooling2D(2)(e3)
    
    e4 = conv_block(p3, FILTERS_BASE * 8)
    if USE_RESIDUAL_BLOCKS:
        e4 = residual_block(e4, FILTERS_BASE * 8)
    e4 = squeeze_excitation_block(e4)
    p4 = MaxPooling2D(2)(e4)
    
    # Bottleneck
    b = conv_block(p4, FILTERS_BASE * 16)
    if USE_RESIDUAL_BLOCKS:
        b = residual_block(b, FILTERS_BASE * 16)
    b = squeeze_excitation_block(b)
    b = Dropout(DROPOUT_RATE)(b)
    
    # Decoder
    u4 = Conv2DTranspose(FILTERS_BASE * 8, 2, strides=2, padding='same')(b)
    if USE_ATTENTION:
        a4 = attention_gate(u4, e4, FILTERS_BASE * 4)
        u4 = Concatenate()([u4, a4])
    else:
        u4 = Concatenate()([u4, e4])
    d4 = conv_block(u4, FILTERS_BASE * 8)
    if USE_RESIDUAL_BLOCKS:
        d4 = residual_block(d4, FILTERS_BASE * 8)
    
    u3 = Conv2DTranspose(FILTERS_BASE * 4, 2, strides=2, padding='same')(d4)
    if USE_ATTENTION:
        a3 = attention_gate(u3, e3, FILTERS_BASE * 2)
        u3 = Concatenate()([u3, a3])
    else:
        u3 = Concatenate()([u3, e3])
    d3 = conv_block(u3, FILTERS_BASE * 4)
    if USE_RESIDUAL_BLOCKS:
        d3 = residual_block(d3, FILTERS_BASE * 4)
    
    u2 = Conv2DTranspose(FILTERS_BASE * 2, 2, strides=2, padding='same')(d3)
    if USE_ATTENTION:
        a2 = attention_gate(u2, e2, FILTERS_BASE)
        u2 = Concatenate()([u2, a2])
    else:
        u2 = Concatenate()([u2, e2])
    d2 = conv_block(u2, FILTERS_BASE * 2)
    if USE_RESIDUAL_BLOCKS:
        d2 = residual_block(d2, FILTERS_BASE * 2)
    
    u1 = Conv2DTranspose(FILTERS_BASE, 2, strides=2, padding='same')(d2)
    if USE_ATTENTION:
        a1 = attention_gate(u1, e1, FILTERS_BASE // 2)
        u1 = Concatenate()([u1, a1])
    else:
        u1 = Concatenate()([u1, e1])
    d1 = conv_block(u1, FILTERS_BASE)
    if USE_RESIDUAL_BLOCKS:
        d1 = residual_block(d1, FILTERS_BASE)
    
    # Output layer
    outputs = Conv2D(3, 1, activation='sigmoid', padding='same')(d1)
    
    model = Model(inputs, outputs, name='ImprovedColorization')
    
    return model

def build_generator_discriminator():
    """Build GAN-style generator and discriminator"""
    
    # Generator (same as U-Net)
    generator = build_unet_model()
    
    # Discriminator
    def build_discriminator():
        inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        
        x = conv_block(inputs, 64, strides=2)
        x = conv_block(x, 128, strides=2)
        x = conv_block(x, 256, strides=2)
        x = conv_block(x, 512, strides=2)
        
        x = GlobalAveragePooling2D()(x)
        x = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, x, name='Discriminator')
    
    discriminator = build_discriminator()
    
    return generator, discriminator

def get_model(model_type='unet'):
    """Get the specified model architecture"""
    if model_type == 'unet':
        return build_unet_model()
    elif model_type == 'gan':
        return build_generator_discriminator()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Custom layers for advanced architectures
class ColorAttention(Layer):
    """Custom attention layer for color channels"""
    
    def __init__(self, **kwargs):
        super(ColorAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        super(ColorAttention, self).build(input_shape)
        
    def call(self, x):
        attention = tf.nn.softmax(tf.matmul(x, self.W), axis=-1)
        return tf.multiply(x, attention)
    
    def compute_output_shape(self, input_shape):
        return input_shape