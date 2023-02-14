from Amoeaba.ImportAll import *
#from src.Amoeaba import ImportAll

def reduction_cell(input_tensor, cell_base_depth, cell_growth_rate, reduction_cell_multiplier, filters_factor,
                   weight_decay, dropout_rate, reduction_cell_groups):
    
    def cell(inputs, filters, strides):
        x = Conv2D(filters * filters_factor, (1, 1), padding='same', use_bias=False,
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters * filters_factor, (3, 3), strides=strides, padding='same',
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters * reduction_cell_multiplier, (1, 1), padding='same',
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=strides, padding='same')(x)
        x = Dropout(dropout_rate)(x)
        return x
    
    filters = cell_base_depth
    cell_outputs = []
    for i in range(reduction_cell_groups):
        strides = 2 if i == 0 else 1
        x = cell(input_tensor, filters, strides)
        cell_outputs.append(x)
        filters *= cell_growth_rate
        
    return cell_outputs
