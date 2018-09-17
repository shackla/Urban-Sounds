import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

# load sound files - loads all of the files in the file_names array
# librosa.load - loads the audio file as a floating point time series...
# ... also returns the sample samp_rate
# then we append(add) the array (X-amplitude vals of wave)...
# to the empty raw_audio array
# this repeats for every file giving us an array of arrays


def load_sound_files(file_paths):
    raw_audio = []
    for fp in file_paths:
        print ('Loading: ', fp)
        X, samp_rate = librosa.load(fp)
        raw_audio.append(X)
    return raw_audio


def plot_waves(labels, raw_audio):
    i = 1
    plt.figure(figsize=(30, 80), dpi=100)
    for n, f in zip(labels, raw_audio):
        plt.subplot(10, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot", x=0.5, y=0.915, fontsize=12)
    plt.show()


def plot_specgram(labels, raw_audio):
    i = 1
    plt.figure(figsize=(30, 80), dpi=100)
    for n, f in zip(labels, raw_audio):
        plt.subplot(10, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram", x=0.5, y=0.915, fontsize=12)
    plt.show()


def plot_log_power_specgram(labels, raw_audio):
    i = 1
    plt.figure(figsize=(30, 80), dpi=100)
    for n, f in zip(labels, raw_audio):
        plt.subplot(10, 1, i)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(f))**2,
                                    ref=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram", x=0.5, y=0.915,
                 fontsize=12)
    plt.show()


file_names = ["MAI/VAD/testfiles/5.wav", "MAI/VAD/testfiles/7.wav",
              "MAI/VAD/testfiles/8.wav", "MAI/VAD/testfiles/9.wav",
              "MAI/VAD/testfiles/13.wav", "MAI/VAD/testfiles/14.wav",
              "MAI/VAD/testfiles/16.wav"]
labels = ["Jackhammer", "Children Playing", "Jackhammer", "Dog Barking",
          "Street Music", "Jackhammer", "Traffic"]

raw_audio = load_sound_files(file_names)

plot_waves(labels, raw_audio)
plot_specgram(labels, raw_audio)
plot_log_power_specgram(labels, raw_audio)
