from Amoeba import ImportAll
#from src.Amoeaba import ImportAll
# Set up the input shape and number of classes
input_shape = (256, 256, 1)
n_classes = 2

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

train_data,validation_data=get_data(train_dir,train_dir)
# Train the model
history=model.fit(train_data, epochs=epochs, verbose=1,validation_data=validation_data)

# Evaluate the model
test_acc = history.history['val_accuracy'][len(history.history['val_accuracy'])-1]
print("Test accuracy:", test_acc)

model.save(model_save_dir)
