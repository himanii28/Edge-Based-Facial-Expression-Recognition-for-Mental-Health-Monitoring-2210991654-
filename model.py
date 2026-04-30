import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation

def build_lightweight_fer_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Builds a lightweight CNN model optimized for edge devices (~1.1M params).
    Uses stacked depthwise-separable convolutions.
    """
    model = Sequential([
        # Block 1 - Standard Convolution to extract initial features
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2 - Depthwise Separable Convolution
        SeparableConv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        SeparableConv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3 - Depthwise Separable Convolution
        SeparableConv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        SeparableConv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected head
        Flatten(),
        Dense(100), # Tuned to reach exactly ~1.10M params overall
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model):
    """
    Compiles the model with categorical cross-entropy loss and Adam optimizer.
    L = -\sum_{i=1}^7 y_i \log(\hat{y}_i)
    """
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    model = build_lightweight_fer_model()
    model.summary()
