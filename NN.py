import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import loadVectors

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
from keras import Sequential, utils, layers, regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

x_train, y_train, x_validation, y_validation, x_test = loadVectors()

y_train = y_train - 1
y_validation = y_validation - 1;

y_train = utils.to_categorical(y_train)
y_validation = utils.to_categorical(y_validation)

model = Sequential([
    layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.02), input_shape=(4096,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.004)),
    layers.Dropout(0.2),
    layers.Dense(29, activation='softmax')
])

model.compile(optimizer="adam", 
              loss='categorical_crossentropy', 
              metrics=['acc'])
              
model.summary()

checkpoint = ModelCheckpoint("Checkpoints/fc_checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
tb_path = os.path.join('tensorboard')
tensorboard = TensorBoard(log_dir=tb_path, histogram_freq=0, write_graph=True, write_images=True, write_grads=True)
lrate = LearningRateScheduler(step_decay)

callbacks = [ModelCheckpoint, early, tensorboard, lrate ]

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_validation, y_validation), callbacks=callbacks, batch_size=32)

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
