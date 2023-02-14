from Amoeba import ImportAll
#from src.Amoeaba import ImportAll

def build_amoeba_net(input_shape, n_classes, stem_fn, cell_base_depth=32, cell_growth_rate=32,
                     reduction_cell_groups=3, reduction_cell_multiplier=3,
                     normal_cell_groups=3, normal_cell_multiplier=3,
                     filters_factor=4, weight_decay=1e-5, dropout_rate=0.5):

    inputs = ImportAll.Input(shape=input_shape)

    x = ImportAll.stem_fn(inputs, filters_factor, weight_decay)

    reduction_outputs = ImportAll.reduction_cell(x, cell_base_depth, cell_growth_rate, reduction_cell_multiplier, filters_factor,
                       weight_decay, dropout_rate, reduction_cell_groups)

    for output in reduction_outputs:
        x = ImportAll.normal_cell(output, cell_base_depth, cell_growth_rate, normal_cell_multiplier, filters_factor,
                        weight_decay, dropout_rate, normal_cell_groups)

    x = ImportAll.layers.GlobalAveragePooling2D(data_format='channels_last')(x)
    x = ImportAll.layers.Dense(n_classes, activation='softmax')(x)

    model = ImportAll.Model(inputs=inputs, outputs=x)
    return model

