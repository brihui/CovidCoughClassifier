import os
import numpy as np
from keras.layers import LSTM
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import layers, models, optimizers, datasets
import tensorflow as tf
from keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, BatchNormalization
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt, pyplot
from matplotlib import image
from sklearn.model_selection import train_test_split


size = 2011
IMG_WIDTH = 1400
IMG_HEIGHT = 500
IMG_DIM = (IMG_HEIGHT, IMG_WIDTH)
HALF_IMG_DIM = tuple(int(ti / 6) for ti in IMG_DIM)


def create_test_val(spectro_path):
    """
    This function reads in the spectrograms given the path to them and creates
    a train/test set to call the appropriate function with.
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

    train_set, test_set, train_label, test_label = train_test_split(train_imgs_scaled, train_labels, test_size=0.3,
                                                                    stratify=train_labels)
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

    train_label = train_label.reshape(-1, 1)
    test_label = test_label.reshape(-1, 1)

    os.chdir(root_directory)

    # Load a positive spectrogram
    # os.chdir(os.getcwd() + "/spectrograms/coughvid")
    # covid_spectro = image.imread('1_922.jpg')
    # covid_spectro_resized = resize(covid_spectro, output_shape=HALF_IMG_DIM)
    # covid_spectro_resized = np.asarray(covid_spectro_resized)
    # print(covid_spectro_resized.shape)
    # print(np.asarray(covid_spectro_resized).shape)

    os.chdir(root_directory)

    # classes = ['Not Covid', 'Covid']
    # plt.figure(2)
    # plt.imshow(train_imgs[0])
    # plt.xlabel(classes[int(train_labels[0])])
    # plt.show()

    lstm_prediction(train_set, test_set, train_label, test_label)
    # lenet_5(train_set, train_label, test_set, test_label)
    # custom_cnn(train_set, test_set, train_label, test_label)
    # resnet_weights(train_set, test_set, train_label, test_label)
    # resnet_prediction(train_set, test_set, train_label, test_label)


def use_test_data(path, threshold):
    """
    Given an absolute path to test spectrograms, call resnet predict.
    :param path: String with path to spectrograms
    """

    root_directory = os.getcwd()

    os.chdir(path)
    test_spectrograms = [file for file in os.listdir()]
    test_images = []
    test_labels = []

    positives = 0
    negatives = 0
    count = 0

    for img in test_spectrograms:
        if img == ".DS_Store":
            pass
        else:
            if img[0] == '0' and negatives < int(size * 0.5):
                negatives += 1
                test_labels.append(int(img[0]))
                image_data = image.imread(img)
                image_data = resize(image_data, output_shape=(166, 466, 3))
                test_images.append(image_data)
                count += 1
            elif img[0] == '1' and positives < int(size * 0.5):
                positives += 1
                test_labels.append(int(img[0]))
                image_data = image.imread(img)
                image_data = resize(image_data, output_shape=(166, 466, 3))
                test_images.append(image_data)
                count += 1
        if count > size:
            break

    test_images = np.array(test_images)

    test_images_scaled = test_images.astype('float32')
    test_images_scaled /= 255

    os.chdir(root_directory)

    # resnet_prediction(test_images_scaled, test_labels)
    # lstm_testing(test_images_scaled, test_labels)
    cnn_testing(test_images_scaled, test_labels)
    return resnet_prediction(test_images_scaled, test_labels, threshold)


def lenet_5(train_set, train_label, test_set, test_label):
    lenet_5 = models.Sequential([
        # Not grayscale like lenet-5
        layers.Conv2D(6, (5, 5), strides=1, activation='relu', input_shape=(166, 466, 3)),
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

    lenet_5.fit(train_set, train_label, epochs=10, batch_size=64)

    print('-' * 100)

    # lenet_5.evaluate(test_set, test_label)
    print(lenet_5.predict(np.asarray([test_set[17]])))
    print(lenet_5.predict(np.asarray([test_set[2]])))
    print(lenet_5.predict(np.asarray([test_set[8]])))
    print(lenet_5.predict(np.asarray([test_set[11]])))
    print(lenet_5.predict(np.asarray([test_set[28]])))


def custom_cnn(train_set, test_set, train_label, test_label):
    custom = models.Sequential([
        # Repeated convolutional layers increase receptive field
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(83, 233, 3)),
        layers.Dropout(0.3),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Dropout(0.3),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(0.3),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Dropout(0.3),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),
        # Multiple dense layers to refine classification voting
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        # Softmax probability function
        layers.Dense(1, activation='sigmoid')
    ])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    custom.compile(optimizer=optimizers.RMSprop(learning_rate=0.00001), loss=tf.keras.losses.binary_crossentropy,
                   metrics=['binary_accuracy'])

    history = custom.fit(train_set, train_label, epochs=50, batch_size=64, callbacks=[callback], validation_split=0.3)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    custom.save(os.getcwd() + '/coswara_custom_model.h5')

    print('-' * 100)

    custom.evaluate(test_set, test_label)

    pred = custom.predict(test_set)
    print(pred)

    print(custom.predict(np.asarray([test_set[17]])))
    print(custom.predict(np.asarray([test_set[2]])))
    print(custom.predict(np.asarray([test_set[8]])))
    print(custom.predict(np.asarray([test_set[11]])))
    print(custom.predict(np.asarray([test_set[28]])))

##
def resnet_weights(train_set, test_set, train_label, test_label):
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

    resnet_model.compile(optimizer=optimizers.RMSprop(learning_rate=0.00001), loss=tf.keras.losses.binary_crossentropy,
                         metrics=['binary_accuracy'])

    resnet_model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    print('-' * 100)

    history = resnet_model.fit(train_set, train_label, epochs=100, batch_size=64, callbacks=[callback])

    print(history)

    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    resnet_model.save(os.getcwd() + '/coswara_resnet_model.h5')

    print(resnet_model.predict(np.asarray([test_set[17]])))
    print(resnet_model.predict(np.asarray([test_set[2]])))
    print(resnet_model.predict(np.asarray([test_set[8]])))
    print(resnet_model.predict(np.asarray([test_set[11]])))
    print(resnet_model.predict(np.asarray([test_set[28]])))


def resnet_prediction(test_data, test_labels, threshold):
    path = os.getcwd() + "/coswara_resnet_model.h5"

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

    resnet_model.load_weights(path)

    prediction = resnet_model.predict(test_data)
    # print(prediction)
    prediction = prediction[:, 0]
    prediction_binary = [0 if pred < threshold else 1 for pred in prediction]
    # prediction_binary = prediction_binary[:, 0]

    # print(prediction_binary)

    accuracy = accuracy_score(test_labels, prediction_binary)
    f1 = f1_score(test_labels, prediction_binary)

    print('Classification Threshold ', threshold)
    print('Accuracy: ', accuracy)
    print('F1 score: ', f1)
    return accuracy, f1


def validate_threshold():
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
    threshold_labels = ['0.2', '0.25', '0.3', '0.35', '0.4']
    accuracy_scores = []
    f1_scores = []
    for threshold in thresholds:
        accuracy, f1 = use_test_data(os.getcwd() + "/spectrograms/coswara", threshold)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)

    plt.plot(threshold_labels, accuracy_scores, label='Accuracy Scores')
    plt.plot(threshold_labels, f1_scores, label='F1 Scores')
    plt.title('Resnet Accuracy and F1 Score at different thresholds')
    plt.xlabel('Thresholds')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.show()


def cnn_testing(test_data, test_labels):
    path = os.getcwd() + "/coswara_custom_model.h5"
    print(path)

    custom = models.Sequential([
        # Repeated convolutional layers increase receptive field
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(83, 233, 3)),
        layers.Dropout(0.3),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Dropout(0.3),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(0.3),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Dropout(0.3),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),
        # Multiple dense layers to refine classification voting
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        # Softmax probability function
        layers.Dense(1, activation='sigmoid')
    ])

    custom.load_weights(path)

    prediction = custom.predict(test_data)
    print(prediction)
    prediction = prediction[:, 0]
    prediction_binary = [0 if pred < 0.5 else 1 for pred in prediction]
    # prediction_binary = prediction_binary[:, 0]

    print(prediction_binary)

    accuracy = accuracy_score(test_labels, prediction_binary)
    f1 = f1_score(test_labels, prediction_binary)

    print('Accuracy: ', accuracy)
    print('F1 score: ', f1)


def lstm_testing(test_data, test_labels):
    path = os.getcwd() + "/coswara_lstm_model.h5"
    print(path)

    test_data = np.dot(test_data[..., :3], [0.2989, 0.5870, 0.1140])

    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(83, 233), dropout=0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.load_weights(path)

    prediction = model.predict(test_data)
    print(prediction)
    prediction = prediction[:, 0]
    prediction_binary = [0 if pred < 0.5 else 1 for pred in prediction]
    # prediction_binary = prediction_binary[:, 0]

    print(prediction_binary)

    accuracy = accuracy_score(test_labels, prediction_binary)
    f1 = f1_score(test_labels, prediction_binary)

    print('Accuracy: ', accuracy)
    print('F1 score: ', f1)


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

    # Normalize RGB values to control weight
    x_train = x_train / 255
    x_test = x_test / 255

    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.5)
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.5)
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.5)

    print(len(x_train))

    resnet_weights(x_train, x_test, y_train, y_test)


def lstm_prediction(train_set, test_set, train_label, test_label):
    train_set = np.dot(train_set[..., :3], [0.2989, 0.5870, 0.1140])
    test_set = np.dot(test_set[..., :3], [0.2989, 0.5870, 0.1140])

    model = Sequential()
    model.add(LSTM(10, activation='relu', dropout=0.3, input_shape=(83, 233)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    # train_set = train_set.reshape((train_set.shape[0], train_set.shape[1], 3))

    history = model.fit(train_set, train_label, epochs=50, callbacks=[callback], validation_split=0.3)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    model.evaluate(test_set, test_label)

    model.save(os.getcwd() + '/coswara_lstm_model.h5')

    # print(model.predict(np.asarray([test_set[17]])))
    # print(model.predict(np.asarray([test_set[2]])))
    # print(model.predict(np.asarray([test_set[8]])))
    # print(model.predict(np.asarray([test_set[11]])))
    # print(model.predict(np.asarray([test_set[28]])))


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # use_test_data(os.getcwd() + "/spectrograms/coswara", 0.4)
    # validate_threshold()
    create_test_val(os.getcwd() + "/spectrograms/coswara")
    use_test_data(os.getcwd() + "/spectrograms/coughvid_test")
    # test_resnet()


if __name__ == "__main__":
    main()
