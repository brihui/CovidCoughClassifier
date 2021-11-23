import os
import matplotlib.pyplot as plt
# for loading and visualizing audio files
import librosa
import librosa.display


# Step 1: Check data
audio_fpath = "audio/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ", len(audio_clips))
index_start = 0
if not audio_clips[0].endswith('.wav'):
    print(audio_clips[0])
    index_start = 1

# Step 2: Load audio file and visualize its waveform (using librosa)
# default sr is 22050
for index in range(index_start, len(audio_clips)):
    print(index)
    x, sr = librosa.load(audio_fpath+audio_clips[index], sr=22050)
    # print(type(x), type(sr))  # <class 'numpy.ndarray'> <class 'int'>
    # print(x.shape, sr)

    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.show()

    # Convert the audio waveform to spectrogram
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()


