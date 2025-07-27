import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D
from keras.optimizers import Adam


def conv_layer(conv_x, filters):
    """Standard convolutional layer for DenseNet"""
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    """Dense block with multiple convolutional layers and dense connections"""
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    """Transition block to reduce feature map size"""
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters


def custom_dense_net(input_shape=(224, 224, 3), filters=64, growth_rate=32, 
                    classes=2, dense_block_size=4, layers_in_block=4):
    """
    Custom DenseNet implementation with configurable parameters
    
    Args:
        input_shape: Input image dimensions (height, width, channels)
        filters: Initial number of filters
        growth_rate: Growth rate for dense connections
        classes: Number of output classes
        dense_block_size: Number of dense blocks
        layers_in_block: Number of layers in each dense block
        
    Returns:
        Keras Model instance
    """
    input_img = Input(shape=input_shape)
    
    # Initial convolution with larger kernel for medical images
    x = Conv2D(filters, (7, 7), strides=(2, 2), kernel_initializer='he_uniform', 
              padding='same', use_bias=False)(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(dense_x)
    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    
    # Create dense blocks with transition layers
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        # Compression in transition layers (using 0.5 compression factor)
        filters = int(filters * 0.5)
        dense_x, filters = transition_block(dense_x, filters)

    # Final dense block (no transition afterward)
    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    
    # Final batch norm and activation
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    
    # Global pooling and classification
    dense_x = GlobalAveragePooling2D()(dense_x)
    
    # Additional fully connected layer for better feature extraction
    dense_x = Dense(256, activation='relu')(dense_x)
    dense_x = Dropout(0.5)(dense_x)
    
    # Output layer
    output = Dense(classes, activation='softmax')(dense_x)

    model = Model(input_img, output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Create model with parameters for chest X-ray classification
model = custom_dense_net(
    input_shape=(224, 224, 3),  # Standard medical image size
    filters=64,                 # Starting filters
    growth_rate=32,             # Standard growth rate for DenseNet
    classes=2,                  # Binary classification (normal vs abnormal)
    dense_block_size=4,         # 4 dense blocks (standard DenseNet architecture)
    layers_in_block=6           # 6 layers per block for more capacity
)

# Example of model usage:
model.summary()

# Define callbacks for training
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_densenet_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7
    )
]