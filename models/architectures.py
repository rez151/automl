import tensorflow as tf
from keras import Sequential
from keras.applications import InceptionV3, Xception, VGG16, VGG19, ResNet50, InceptionResNetV2, MobileNet, DenseNet121, \
    DenseNet169, DenseNet201
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout


def load_alexnet(width, height, classes_num):
    # AlexNet with batch normalization in Keras
    # input image is 224x224

    # Define the Model

    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(
            Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same', input_shape=(width, height, 3)))
        # for original Alexnet
        # model.add(Conv2D(96, (3,3), strides=(2,2), activation='relu', padding='same', input_shape=(img_height, img_width, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Local Response normalization for Original Alexnet
        model.add(BatchNormalization())

        model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # Local Response normalization for Original Alexnet
        model.add(BatchNormalization())

        model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # Local Response normalization for Original Alexnet
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(4096, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(4096, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(classes_num, activation='softmax'))

    return model


def load_customnet(width, height, classes_num, layer, min_filters, max_filters, conv_window=3, pool=3, strides=2,
                   dense_layers=2,
                   dense_units=256,
                   dropout=0.1):
    with tf.device('/cpu:0'):

        hidden_layers = 0

        model = Sequential()
        model.add(
            Conv2D(min_filters, (11, 11), strides=4, activation='relu', padding='same', input_shape=(width, height, 3)))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())

        filters = min_filters + 32

        model.add(Conv2D(filters, conv_window, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=pool, strides=strides))

        filters = min_filters + 32
        while filters <= max_filters:
            model.add(Conv2D(filters, conv_window, activation='relu', padding='same'))
            model.add(BatchNormalization())
            filters += 32
            hidden_layers += 1

        if hidden_layers < layer:

            for i in range(hidden_layers, layer):
                model.add(Conv2D(filters, conv_window, activation='relu', padding='same'))
                model.add(BatchNormalization())

        model.add(Conv2D(filters, conv_window, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=pool, strides=strides))
        model.add(BatchNormalization())

        model.add(Flatten())

        for i in range(dense_layers):
            model.add(Dense(dense_units, activation='tanh'))
            model.add(Dropout(dropout))

        model.add(Dense(classes_num, activation='softmax'))

        return model


def load_inception_v3(width, height, classes_num):
    with tf.device('/cpu:0'):
        model = InceptionV3(weights=None, input_shape=(width, height, 3), classes=classes_num)

    return model


def load_xception(width, height, classes_num):
    with tf.device('/cpu:0'):
        model = Xception(weights=None, input_shape=(width, height, 3), classes=classes_num)

    return model


def load_vgg16(width, height, classes_num):
    with tf.device('/cpu:0'):
        model = VGG16(weights=None, input_shape=(width, height, 3), classes=classes_num)

    return model


def load_vgg19(width, height, classes_num):
    with tf.device('/cpu:0'):
        model = VGG19(weights=None, input_shape=(width, height, 3), classes=classes_num)

    return model


def load_resnet50(width, height, classes_num):
    with tf.device('/cpu:0'):
        model = ResNet50(weights=None, input_shape=(width, height, 3), classes=classes_num)

    return model


def load_inceptionresnet_v2(width, height, classes_num):
    with tf.device('/cpu:0'):
        model = InceptionResNetV2(weights=None, input_shape=(width, height, 3), classes=classes_num)

    return model


def load_mobilenet(width, height, classes_num):
    with tf.device('/cpu:0'):
        model = MobileNet(weights=None, input_shape=(width, height, 3), classes=classes_num)

    return model


def load_densenet121(width, height, classes_num):
    with tf.device('/cpu:0'):
        model = DenseNet121(weights=None, input_shape=(width, height, 3), classes=classes_num)

    return model


def load_densenet169(width, height, classes_num):
    with tf.device('/cpu:0'):
        model = DenseNet169(weights=None, input_shape=(width, height, 3), classes=classes_num)

    return model


def load_densenet201(width, height, classes_num):
    with tf.device('/cpu:0'):
        model = DenseNet201(weights=None, input_shape=(width, height, 3), classes=classes_num)

    return model
