from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import Input

def stem(inputs):
    # First convolutional layer
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Two more convolutional layers
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Max pooling layer
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    
    return x
