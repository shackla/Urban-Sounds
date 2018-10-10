import os
import librosa
import numpy as np
import pylab
import matplotlib
import librosa.display
import csv

parent_dir = "/home/alex/Documents/MAI/VAD/UrbanSounds/spectrumfiles"
audio_dir = "wavfiles"
sub_dirs = ["training", "validation"]

IDnums = ["train_IDs.csv", "val_IDs.csv"]

matplotlib.use('Agg')


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


for i in range(0, 1):
    csv_path = os.path.join(parent_dir, IDnums[i])
    IDs = readcsv(csv_path)

    data_length = len(IDs)

    for j in range(0, data_length-1):
        file_path = os.path.join(parent_dir, audio_dir, str(IDs[j])+".wav")
        img_save = os.path.join(parent_dir, sub_dirs[i], str(IDs[j])+".jpg")
        try:
            X, sample_rate = librosa.load(file_path)
            pylab.axis('off')
            S = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            pylab.savefig(img_save, bbox_inches=None, pad_inches=0)
            print(img_save, "success")

        except Exception as e:
            print('Error encountered while parsing the file:', file_path)
