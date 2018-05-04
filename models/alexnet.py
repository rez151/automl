from keras import Sequential, optimizers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout


def getAlexnet(img_width, img_height, classes_num, lr):
    # AlexNet with batch normalization in Keras
    # input image is 224x224

    # Define the Model
    model = Sequential()
    model.add(
        Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same', input_shape=(img_height, img_width, 3)))
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
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=lr),
                  metrics=['accuracy'])

    return model
