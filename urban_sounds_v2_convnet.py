import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
import matplotlib as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import io
import csv

parent_dir = '/home/alex/Documents/MAI/VAD/UrbanSounds/spectrumfiles'
tr_img_dir = 'training'
ts_img_dir = 'validation'
csv_tr_file = 'training_labels/training_labels.csv'
csv_ts_file = 'validation_labels/validation_labels.csv'

train_dir = os.path.join(parent_dir, tr_img_dir)
validation_dir = os.path.join(parent_dir, ts_img_dir)

csv_path_tr = os.path.join(parent_dir, csv_tr_file)
csv_path_ts = os.path.join(parent_dir, csv_ts_file)

train_labels = []
with open(csv_path_tr) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        train_labels.append(row)

test_labels = []
with open(csv_path_ts) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        test_labels.append(row)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(480, 640),
    batch_size=64,
    class_mode='binary')


validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(480, 640),
    batch_size=64,
    class_mode='binary')


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),
                        activation='relu', input_shape=(480, 640, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=(validation_generator),
                              validation_steps=50)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(loss) + 1)

plt.figure(1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
