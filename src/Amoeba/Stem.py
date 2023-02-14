from Amoeba.ImportAll import *
#from src.Amoeaba import ImportAll

#from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D

def stem_fn(input_tensor, filters_factor, weight_decay):
    x = ImportAll.Conv2D(32 * filters_factor, (3, 3), strides=(2, 2), padding='valid',
               use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input_tensor)
    x = ImportAll.BatchNormalization()(x)
    x = ImportAll.Activation('relu')(x)
    x = ImportAll.Conv2D(32 * filters_factor, (3, 3), padding='valid', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = ImportAll.BatchNormalization()(x)
    x = ImportAll.Activation('relu')(x)
    x = ImportAll.Conv2D(64 * filters_factor, (3, 3), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = ImportAll.BatchNormalization()(x)
    x = ImportAll.Activation('relu')(x)
    x = ImportAll.Conv2D(64 * filters_factor, (3, 3), padding='valid', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = ImportAll.BatchNormalization()(x)
    x = ImportAll.Activation('relu')(x)
    x = ImportAll.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    return x
