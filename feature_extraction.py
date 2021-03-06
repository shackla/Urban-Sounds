import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

parent_dir = "/home/alex/Documents/MAI/VAD/UrbanSounds/audiofiles"
tr_sub_dirs = "Train"
ts_sub_dirs = "Test"

train = pd.read_csv(os.path.join(parent_dir, 'train.csv'))
validate = pd.read_csv(os.path.join(parent_dir, 'test.csv'))


def parser(row):
    file_name = os.path.join(parent_dir, ts_sub_dirs, str(row.ID)+'.wav')

    try:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                                             n_mfcc=40).T, axis=0)
    except Exception as e:
        print('Error encountered while parsing the file:', file_name)

        return 'None', 'None'
    feature = mfccs
    label = row.Class
    print(file_name)
    print(label)
    return pd.Series([feature, label], index=['feature', 'label'])


def parser_Val(row):
    file_name = os.path.join(parent_dir, tr_sub_dirs, str(row.ID)+'.wav')

    try:
        val_X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        mfccs = np.mean(librosa.feature.mfcc(y=val_X, sr=sample_rate,
                                             n_mfcc=40).T, axis=0)
    except Exception as e:
        print('Error encountered while parsing the file:', file_name)

        return 'None', 'None'
    feature = mfccs
    label = row.Class
    print(file_name)
    print(label)
    return pd.Series([feature, label], index=['feature', 'label'])


temp = train.apply(parser_Val, axis=1)
temp_Val = validate.apply(parser, axis=1)


temp.columns = ['feature', 'label']
temp_Val.columns = ['feature', 'label']


X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())
val_X = np.array(temp_Val.feature.tolist())
val_Y = np.array(temp_Val.label.tolist())


print(X.shape, X)
print(val_X.shape, val_X)


label_encoder = LabelEncoder()
y = np_utils.to_categorical(label_encoder.fit_transform(y))

lb2 = LabelEncoder()
val_Y = np_utils.to_categorical(lb2.fit_transform(val_Y))

print(y.shape, y)
print(val_Y.shape, val_Y)

num_labels = val_Y.shape[1]
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


history = model.fit(X, y, batch_size=32, epochs=15,
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
