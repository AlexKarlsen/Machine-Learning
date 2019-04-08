import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import loadVectors

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
from tensorflow import keras

x_train, y_train, x_validation, y_validation, x_test = loadVectors()

y_train = y_train - 1
y_validation = y_validation - 1;

y_train = keras.utils.to_categorical(y_train)
y_validation = keras.utils.to_categorical(y_validation)

model = keras.Sequential([
    keras.layers.Dense(4096, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02), input_shape=(4096,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.004)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(29, activation='softmax')
])

model.compile(optimizer="adam", 
              loss='categorical_crossentropy', 
              metrics=['acc'])
              
model.summary()

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_validation, y_validation))

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
