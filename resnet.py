import os
import numpy as np
from tensorflow.keras import layers, models
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.applications.resnet import ResNet50
from keras.models import Model
import keras
from matplotlib import pyplot as plt
from matplotlib import image
from PIL import Image
from sklearn.model_selection import train_test_split


def create_resnet_weights(spectro_path):
    """
    THis function creates the resnet weights given the path to the spectrograms
    :param spectro_path: String of absolute path to the spectrogram folder
    """

    # Get list of healthy and list of covid spectrograms
    os.chdir(spectro_path)
    train_spectrograms = [file for file in os.listdir()]

    # # load_img can take in tuple of image size to resize the image
    # train_imgs = [image.imread(img) for img in train_spectrograms if img != ".DS_Store"]
    count = 0
    train_imgs = []
    for img in train_spectrograms:
        if img == ".DS_Store":
            pass
        else:
            image_data = image.imread(img)
            train_imgs.append(image_data)
            count +=1
        if count > 999:
            break
    train_imgs = np.array(train_imgs)
    # Take the first character of file name for label, 0 = healthy 1 = covid
    # train_labels = [file_name[0] for file_name in train_spectrograms if file_name != ".DS_Store"]
    train_labels = []
    count = 0
    for name in train_spectrograms:
        if name == ".DS_Store":
            pass
        else:
            train_labels.append(int(name[0]))
            count += 1
        if count > 999:
            break

    print('Train dataset shape: ', train_imgs.shape)

    # Scale images from 0-255 to 0-1
    train_imgs_scaled = train_imgs.astype('float32')
    train_imgs_scaled /= 255

    train_set, test_set, train_label, test_label = train_test_split(train_imgs_scaled, train_labels, test_size=0.3)
    train_label = np.asarray(train_label)
    test_label = np.asarray(test_label)
    print(train_label)
    print(test_label[17])
    print(test_label[2])
    print(test_label[8])
    print(test_label[11])
    print(test_label[28])

    print(train_set.shape)
    print(test_set.shape)
    print(train_label.shape)
    print(test_label.shape)

    train_label = train_label.reshape(-1,1)
    test_label = test_label.reshape(-1,1)

    lenet_5(train_set, train_label, test_set, test_label)

    classes = ['Not Covid', 'Covid']
    # plt.imshow(train_imgs[0])
    # plt.xlabel(classes[int(train_labels[0])])
    # plt.show()


def lenet_5(train_set, train_label, test_set, test_label):
    lenet_5 = models.Sequential([
        # Not grayscale like lenet-5
        layers.Conv2D(6, (5, 5), strides=1, activation='relu', input_shape=(500, 1400, 3)),
        layers.AvgPool2D((2, 2), strides=2),
        layers.Conv2D(16, (5, 5), strides=1, activation='relu'),
        layers.AvgPool2D((2, 2), strides=2),
        layers.Conv2D(120, (5, 5), strides=1, activation='relu'),
        # Shape automatically resolved
        layers.Flatten(),
        layers.Dense(84, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Sparse because we are using index rather than flat matrix for class representation
    lenet_5.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

    lenet_5.fit(train_set, train_label, epochs=1)

    print('-' * 100)

    # lenet_5.evaluate(test_set, test_label)
    print(lenet_5.predict(np.asarray([test_set[17]])))
    print(lenet_5.predict(np.asarray([test_set[2]])))
    print(lenet_5.predict(np.asarray([test_set[8]])))
    print(lenet_5.predict(np.asarray([test_set[11]])))
    print(lenet_5.predict(np.asarray([test_set[28]])))


def custom_cnn(train_set, train_label, test_set, test_label):
    custom = models.Sequential([
        # Repeated convolutional layers increase receptive field
        layers.Conv2D(12, (4, 4), strides=1, activation='relu'),
        layers.Conv2D(24, (4, 4), strides=1, activation='relu'),
        # Down sampling with stride can be cheaper as you convolve + downsample at the same time
        layers.Conv2D(48, (4, 4), strides=2, activation='relu'),
        layers.Conv2D(192, (4, 4), strides=2, activation='relu'),
        # Maxpool retains most prominent features
        # layers.MaxPool2D((2, 2), strides=2),

        layers.Flatten(),
        # Multiple dense layers to refine classification voting
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        # Softmax probability function
        layers.Dense(1, activation='softmax')
    ])

    custom.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

    custom.fit(train_set, train_label, epochs=1)

    print('-' * 100)

    print(custom.predict(np.asarray([test_set[69]])))
    print(custom.predict(np.asarray([test_set[17]])))
    print(custom.predict(np.asarray([test_set[2]])))
    print(custom.predict(np.asarray([test_set[8]])))
    print(custom.predict(np.asarray([test_set[11]])))
    print(custom.predict(np.asarray([test_set[28]])))


def main():
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         tf.config.set_visible_devices(gpus[0], 'GPU')
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     except RuntimeError as e:
    #         # Visible devices must be set before GPUs have been initialized
    #         print(e)

    create_resnet_weights(os.getcwd() + "/spectrograms/coswara")


if __name__ == "__main__":
    main()
