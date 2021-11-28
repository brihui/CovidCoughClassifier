import os
import numpy as np
from tensorflow.keras import layers, models, optimizers, datasets
import tensorflow as tf
from keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from matplotlib import image
from sklearn.model_selection import train_test_split


size = 100
IMG_WIDTH = 1400
IMG_HEIGHT = 500
IMG_DIM = (IMG_HEIGHT, IMG_WIDTH)


def create_resnet_weights(spectro_path):
    """
    THis function creates the resnet weights given the path to the spectrograms
    :param spectro_path: String of absolute path to the spectrogram folder
    """
    root_directory = os.getcwd()

    # Get list of healthy and list of covid spectrograms
    os.chdir(spectro_path)
    train_spectrograms = [file for file in os.listdir()]

    # # load_img can take in tuple of image size to resize the image
    # train_imgs = [image.imread(img) for img in train_spectrograms if img != ".DS_Store"]
    count = 0
    train_imgs = []
    HALF_IMG_DIM = tuple(int(ti / 4) for ti in IMG_DIM)
    positives = 0
    negatives = 0
    count = 0
    train_imgs = []
    train_labels = []
    for img in train_spectrograms:
        if img == ".DS_Store":
            pass
        else:
            if img[0] == '0' and negatives < int(size * 0.7):
                negatives += 1
                train_labels.append(int(img[0]))
                image_data = image.imread(img)
                image_data = resize(image_data, output_shape=HALF_IMG_DIM)
                train_imgs.append(image_data)
                count += 1
            elif img[0] == '1' and positives < int(size * 0.3):
                positives += 1
                train_labels.append(int(img[0]))
                image_data = image.imread(img)
                image_data = resize(image_data, output_shape=HALF_IMG_DIM)
                train_imgs.append(image_data)
                count += 1
        if count > size:
            break
    train_imgs = np.array(train_imgs)
    print(train_imgs.shape)
    # for img in train_spectrograms:
    #     if img == ".DS_Store":
    #         pass
    #     else:
    #         image_data = image.imread(img)
    #         image_data = resize(image_data, output_shape=HALF_IMG_DIM)
    #         train_imgs.append(image_data)
    #         count += 1
    #     if count > 999:
    #         break
    # train_imgs = np.array(train_imgs)
    # Take the first character of file name for label, 0 = healthy 1 = covid
    # train_labels = [file_name[0] for file_name in train_spectrograms if file_name != ".DS_Store"]
    # train_labels = []
    # count = 0
    # for name in train_spectrograms:
    #     if name == ".DS_Store":
    #         pass
    #     else:
    #         train_labels.append(int(name[0]))
    #         count += 1
    #     if count > 999:
    #         break

    print('Train dataset shape: ', train_imgs.shape)

    # Scale images from 0-255 to 0-1
    train_imgs_scaled = train_imgs.astype('float32')
    train_imgs_scaled /= 255

    train_set, test_set, train_label, test_label = train_test_split(train_imgs_scaled, train_labels, test_size=0.3)
    train_label = np.asarray(train_label)
    test_label = np.asarray(test_label)
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

    os.chdir(root_directory)

    # Load a positive spectrogram
    os.chdir(os.getcwd() + "/spectrograms/coughvid")
    covid_spectro = image.imread('1_922.jpg')
    covid_spectro_resized = resize(covid_spectro, output_shape=HALF_IMG_DIM)
    covid_spectro_resized = np.asarray(covid_spectro_resized)
    print(covid_spectro_resized.shape)
    print(np.asarray(covid_spectro_resized).shape)

    os.chdir(root_directory)

    classes = ['Not Covid', 'Covid']
    # plt.imshow(train_imgs[0])
    # plt.xlabel(classes[int(train_labels[0])])
    # plt.show()

    # lenet_5(train_set, train_label, test_set, test_label, covid_spectro_resized)
    # custom_cnn(train_set, train_label, test_set, test_label, covid_spectro_resized)
    # resnet_weights(train_set, test_set, train_label, test_label)
    resnet_prediction(train_set, test_set, train_label, test_label)


def lenet_5(train_set, train_label, test_set, test_label, covid_spectro):
    lenet_5 = models.Sequential([
        # Not grayscale like lenet-5
        layers.Conv2D(6, (5, 5), strides=1, activation='relu', input_shape=(250, 700, 3)),
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
    lenet_5.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['binary_accuracy'])

    lenet_5.fit(train_set, train_label, epochs=1)

    print('-' * 100)

    # lenet_5.evaluate(test_set, test_label)
    print(lenet_5.predict(np.asarray([covid_spectro])))
    print(lenet_5.predict(np.asarray([test_set[17]])))
    print(lenet_5.predict(np.asarray([test_set[2]])))
    print(lenet_5.predict(np.asarray([test_set[8]])))
    print(lenet_5.predict(np.asarray([test_set[11]])))
    print(lenet_5.predict(np.asarray([test_set[28]])))


def custom_cnn(train_set, train_label, test_set, test_label, covid_spectro):
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
        layers.Dense(1, activation='sigmoid')
    ])

    custom.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), loss=tf.keras.losses.binary_crossentropy, metrics=['binary_accuracy'])

    custom.fit(train_set, train_label, epochs=1)

    print('-' * 100)

    print(custom.predict(np.asarray([covid_spectro])))
    print(custom.predict(np.asarray([test_set[17]])))
    print(custom.predict(np.asarray([test_set[2]])))
    print(custom.predict(np.asarray([test_set[8]])))
    print(custom.predict(np.asarray([test_set[11]])))
    print(custom.predict(np.asarray([test_set[28]])))


def resnet_weights(train_set, test_set, train_label, test_label):
    # restnet = ResNet50(include_top=False, weights=None, input_shape=(32, 32, 3))
    # output = restnet.layers[-1].output
    # output = layers.Flatten()(output)
    # restnet = Model(restnet.input, outputs=output)
    # for layer in restnet.layers:
    #     layer.trainable = False
    # restnet.summary()

    base_model_2 = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=10)

    model_2 = Sequential()
    # Add the Dense layers along with activation and batch normalization
    model_2.add(base_model_2)
    model_2.add(Flatten())

    # Add the Dense layers along with activation and batch normalization
    model_2.add(Dense(4000, activation=('relu'), input_dim=512))
    model_2.add(Dense(2000, activation=('relu')))
    model_2.add(Dropout(.4))
    model_2.add(Dense(1000, activation=('relu')))
    model_2.add(Dropout(.3))  # Adding a dropout layer that will randomly drop 30% of the weights
    model_2.add(Dense(500, activation=('relu')))
    model_2.add(Dropout(.2))
    model_2.add(Dense(10, activation=('softmax')))  # This is the classification layer

    # model = Sequential()
    # model.add(restnet)
    #
    # model.add(layers.Flatten())
    #
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(10, activation='softmax'))
    # model.compile(loss='binary_crossentropy',
    #               optimizer=optimizers.RMSprop(learning_rate=2e-5),
    #               metrics=['binary_accuracy'])
    model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_2.summary()

    model_2.fit(train_set, train_label, epochs=5)

    model_2.evaluate(test_set, test_label)

    # model.save(os.getcwd() + '/coswara_cnn_model.h5')

    # print(model.predict(np.asarray([test_set[17]])))


def resnet_prediction(test_set, train_set, train_label, test_label):
    path = os.getcwd() + "/coswara_cnn_restnet50.h5"
    print(path)
    restnet = ResNet50(include_top=False, input_shape=(250, 700, 3))
    restnet.load_weights(path, by_name=True)
    output = restnet.layers[-1].output
    output = layers.Flatten()(output)
    restnet = Model(restnet.input, outputs=output)
    for layer in restnet.layers:
        layer.trainable = False

    # restnet.summary()

    model = Sequential()
    model.add(restnet)
    model.add(Dense(512, activation='relu', input_dim=(250, 700, 3)))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=2e-5),
                  metrics=['binary_accuracy'])

    print(model.predict(np.asarray([test_set[17]])))
    print(model.predict(np.asarray([test_set[2]])))
    print(model.predict(np.asarray([test_set[8]])))
    print(model.predict(np.asarray([test_set[11]])))
    print(model.predict(np.asarray([test_set[28]])))


def test_resnet():
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    # Flattens y training data for usability
    y_train = y_train.reshape(-1, )
    # print(x_train)

    # View training image with label
    # plt.imshow(x_train[9])
    # plt.xlabel(classes[y_train[9]])
    # plt.show()
    # plt.close('all')

    # Normalize RGB values to control weights
    x_train = x_train / 255
    x_test = x_test / 255

    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.5)
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.5)
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.5)

    print(len(x_train))

    resnet_weights(x_train, x_test, y_train, y_test)


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

    # create_resnet_weights(os.getcwd() + "/spectrograms/coswara")
    test_resnet()


if __name__ == "__main__":
    main()
