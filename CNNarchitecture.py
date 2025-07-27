model = models.Sequential([
    layers.Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (224, 224,3)),
    layers.BatchNormalization(),
    layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    layers.Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    layers.Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    layers.Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    layers.Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'),
    layers.Flatten(),
    layers.Dense(units = 128 , activation = 'relu'),
    layers.Dropout(0.2),
    layers.Dense(units = 2 , activation = 'softmax')
])
model.compile(optimizer = "adam" , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
callbacks = [ModelCheckpoint('.mdl_wts.hdf5', monitor='val_loss', save_best_only=True), EarlyStopping(monitor='val_loss', patience=5)]
model.summary()