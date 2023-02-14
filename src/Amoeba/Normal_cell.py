from Amoeba import ImportAll
#from src.Amoeaba import ImportAll

def normal_cell(reduction_outputs, cell_base_depth, cell_growth_rate, normal_cell_multiplier, filters_factor, weight_decay, dropout_rate, normal_cell_groups):
    def build_cell(input_tensor, filters, reduction, stride=1):
        x = Conv2D(filters, (3, 3), padding='same', strides=stride, use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if reduction:
            x = Conv2D(filters, (3, 3), padding='same', strides=stride, use_bias=False,
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

        return x

    inputs = reduction_outputs[:2] # first two tensors as inputs
    input_tensor = Concatenate()(inputs) # concatenate inputs along channel axis
    filters = int(cell_base_depth * filters_factor)
    x = build_cell(input_tensor, filters, True) # reduction=True for first cell
    for i in range(normal_cell_groups - 1):
        x = build_cell(x, filters, False)

    return x
