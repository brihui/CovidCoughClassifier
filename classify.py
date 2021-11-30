import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, BatchNormalization
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from matplotlib import image


def is_covid(file_name):
    """
    Given a file name of a file in the cough folder, convert it to a spectrogram,
    and classify it.
    :param file_name: String
    :return: Boolean
    """
    # Get path to the coughs
    root_directory = os.getcwd()
    cough_path = os.getcwd() + '/coughs_to_classify/'

    # If wav, convert this way
    if file_name[-4:] == ".wav":
        x, sr = librosa.load(cough_path + file_name, sr=22050)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        # plt.show()
        plt.savefig(cough_path + file_name[:-4] + ".png")
        plt.close()

        os.chdir(cough_path)

        image_data = image.imread(file_name[:-4] + ".png")
        image_data = resize(image_data, output_shape=(166, 466, 3))

        image_data = np.asarray(image_data)
        image_data /= 255
    elif file_name[-4:] == ".jpg":
        os.chdir(cough_path)

        image_data = image.imread(file_name)
        image_data = resize(image_data, output_shape=(166, 466, 3))

        image_data = np.asarray(image_data)
        image_data /= 255

    os.chdir(root_directory)

    weights_path = os.getcwd() + "/coswara_resnet_model.h5"

    resnet_model = Sequential()

    pretrained_model = ResNet50(include_top=False, input_shape=(166, 466, 3), weights='imagenet')

    output = pretrained_model.layers[-1].output
    output = Flatten()(output)
    pretrained_model = Model(pretrained_model.input, outputs=output)

    for layer in pretrained_model.layers:
        layer.trainable = False

    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(BatchNormalization())
    resnet_model.add(Dense(256, activation='relu'))
    resnet_model.add(Dropout(0.3))
    resnet_model.add(BatchNormalization())
    resnet_model.add(Dense(128, activation='relu'))
    resnet_model.add(Dropout(0.3))
    resnet_model.add(BatchNormalization())
    resnet_model.add(Dense(64, activation='relu'))
    resnet_model.add(Dropout(0.3))
    resnet_model.add(BatchNormalization())
    resnet_model.add(Dense(1, activation='sigmoid'))

    resnet_model.load_weights(weights_path)

    prediction = resnet_model.predict(np.asarray([image_data]))

    if prediction[0] < 0.3:
        return False
    else:
        return True


if __name__ == "__main__":
    file_name = 'covid5.jpg'
    covid = is_covid(file_name)
    if covid:
        print(f'{file_name} C^3 Test Results: Positive')
    else:
        print(f'{file_name} C^3 Test Results: Negative (Yay)')
