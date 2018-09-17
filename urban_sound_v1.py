import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib as plt
from feature_extraction import temp, temp_Val
import os

os.system('python feature_extraction.py')
temp.columns = ['feature', 'label']
temp_Val.columns = ['feature', 'label']


X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())
val_X = np.array(temp_Val.feature.tolist())
val_Y = np.array(temp_Val.label.tolist())

label_encoder = LabelEncoder()
lb2 = LabelEncoder()

y = np_utils.to_categorical(label_encoder.fit_transform(y))
val_Y = np_utils.to_categorical(lb2.fit_transform(val_Y))

num_labels = y.shape[1]
filter_size = 2


model = Sequential()
model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer='adam')


history = model.fit(X, y, batch_size=32, epochs=5,
                    validation_data=(val_X, val_Y))

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
