# Import dependenciea
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential, utils, layers, regularizers, optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau

# Importing helper scripts
from load_data import loadVectors

# Creating a tensorflow session, that does not allow script to exhaust memory
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .5
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

# Loading locale data
x_train, y_train, x_validation, y_validation, x_test = loadVectors()

# Subtracting class offset
y_train = y_train - 1
y_validation = y_validation - 1;

# Converting to 'one-hot' encoding
y_train = utils.to_categorical(y_train)
y_validation = utils.to_categorical(y_validation)

# Constructing the model
model = Sequential([
    layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.04), input_shape=(4096,)),
    layers.Dropout(0.5),
    layers.Dense(28, activation='relu', kernel_regularizer=regularizers.l2(0.02)),
    layers.Dropout(0.4),
    layers.Dense(29, activation='softmax')
])

# Compiling model by specifying optimizer and loss function
model.compile(optimizer=optimizers.Adam(lr=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['acc'])

# Print Model Summary              
model.summary()

# Define model callbacks
checkpoint = ModelCheckpoint("checkpoints/fc_checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
#tb_path = os.path.join('tensorboard')
#tensorboard = TensorBoard(log_dir=tb_path, histogram_freq=0, write_graph=True, write_images=True, write_grads=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001, verbose=1)
callbacks = [checkpoint, early, reduce_lr ]

# Train the model 
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_validation, y_validation), callbacks=callbacks, batch_size=32, verbose=2)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8), dpi=160)
plt.suptitle('Feature Extraction (no Augmentation)')
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()
