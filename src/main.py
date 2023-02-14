from Amoeaba import ImportAll
#from src.Amoeaba import ImportAll
# Set up the input shape and number of classes
input_shape = (32, 32, 3)
n_classes = 10

# Set up the hyperparameters
stem_multiplier = 3
cell_base_depth = 32
cell_growth_rate = 32
reduction_cell_groups = 3
reduction_cell_multiplier = 3
normal_cell_groups = 3
normal_cell_multiplier = 3
filters_factor = 4
weight_decay = 1e-5
dropout_rate = 0.5
learning_rate = 1e-3
batch_size = 128
epochs = 100

# Build the model
model = ImportAll.build_amoeba_net(input_shape, n_classes, stem_multiplier, cell_base_depth, cell_growth_rate,
                         reduction_cell_groups, reduction_cell_multiplier, normal_cell_groups,
                         normal_cell_multiplier, filters_factor, weight_decay, dropout_rate)

# Compile the model
optimizer = ImportAll.optimizers.Adam(learning_rate=learning_rate)
loss = ImportAll.losses.CategoricalCrossentropy()
metrics = [ImportAll.metrics.CategoricalAccuracy()]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

train_dir=""
model_save_dir=""
train_data=
test_data=
# Train the model
model.fit(data_augmentation(x_train), y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

model.save(model_save_dir)
