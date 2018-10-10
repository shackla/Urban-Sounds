import os
import matplotlib
import pylab
import librosa
import librosa.display
import numpy as np
import csv

matplotlib.use('Agg')
parent_dir = '/home/alex/Documents/MAI/VAD/UrbanSounds/spectrumfiles'
img_dir = 'training'
audio_dir = 'wavfiles'
train_files = 'train_ids2.csv'


def readcsv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=";")

    rownum = 0
    a = []

    for row in reader:
        a.extend(row)
        rownum += 1

    ifile.close()
    return a


csv_path = os.path.join(parent_dir, train_files)

ID = readcsv(csv_path)
max = len(ID)
for i in range(0, max):

    file_name = str(ID[i])
    audio_path = os.path.join(parent_dir, audio_dir, file_name+'.wav')
    save_path = os.path.join(parent_dir, img_dir, file_name+'.jpg')
    check = os.path.exists(save_path)
    if check is True:
        print("Already converted!")
    else:
        sig, fs = librosa.load(audio_path)
        pylab.axis('off')
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        S = librosa.feature.melspectrogram(y=sig, sr=fs)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()
        print(audio_path, " has been succesfully converted")
