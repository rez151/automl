import os

from keras import callbacks
from keras import optimizers
from keras.callbacks import History
from keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

train_data_path = './datasets/train'
test_data_path = './datasets/test'
validation_data_path = './datasets/validation'

"""
Parameters
"""
img_width, img_height = 150, 150
batch_size = 32
samples_per_epoch = 1000
validation_steps = 10
classes_num = 5
lr = 0.00001
epochs = 50

class_weight = {
    0: 3.,
    1: 3.,
    2: 1.,
    3: 3.,
    4: 3.
}

"""
CNN Properties
"""
nb_filters1 = 64
nb_filters2 = 128
conv1_size = 3
conv2_size = 2
conv3_size = 2
conv4_size = 2
conv5_size = 2

pool_size = 2

"""
Model
"""

model = Sequential()
model.add(Convolution2D(nb_filters1, (conv1_size, conv1_size), padding="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, (conv2_size, conv2_size), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, (conv2_size, conv2_size), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, (conv3_size, conv3_size), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, (conv3_size, conv3_size), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, (conv4_size, conv4_size), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, (conv4_size, conv4_size), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, (conv4_size, conv4_size), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=1))

model.add(Convolution2D(nb_filters2, (conv5_size, conv5_size), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=1))




model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.3))
model.add(Dense(classes_num, activation='softmax'))
#model.add(Flatten())
#model.add(Dense(4096, init='normal'))
#model.add(Activation('relu'))
#model.add(Dense(4096, init='normal'))
#model.add(Activation('relu'))
#model.add(Dense(1000, init='normal'))
#model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=lr),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

"""
Tensorboard log
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
# cbks = [tb_cb]

hist = History()
cbks = [hist]

model.fit_generator(
    generator=train_generator,
    samples_per_epoch=samples_per_epoch,
    epochs=epochs,
    callbacks=cbks,
    steps_per_epoch=31,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weight
)

test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator, steps=nb_samples)

print(predict)

target_dir = './models/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
model.save('./models/model20000e.h5')
model.save_weights('./models/weights20000e.h5')

results = hist.history.items()

print(results)

with open("results.txt", "w") as text_file:
    print(results, file=text_file)
