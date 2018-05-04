import os

from keras import callbacks
from keras import optimizers
from keras.callbacks import History
from keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization, Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

train_data_path = './datasets/filtered/train'
test_data_path = './datasets/test'
validation_data_path = './datasets/filtered/validation'

"""
Parameters
"""
img_width, img_height = 500, 500
batch_size = 32
samples_per_epoch = 1000
validation_steps = 10
classes_num = 5
lr = 0.00005
epochs = 2000

class_weight = {
    0: 11.,
    1: 5.,
    2: 1.,
    3: 40.,
    4: 30.
}

"""
Model
"""

#AlexNet with batch normalization in Keras
#input image is 224x224

# Define the Model
model = Sequential()
model.add(Conv2D(96, (11,11), strides=(4,4), activation='relu', padding='same', input_shape=(img_height, img_width, 3)))
# for original Alexnet
#model.add(Conv2D(96, (3,3), strides=(2,2), activation='relu', padding='same', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
# Local Response normalization for Original Alexnet
model.add(BatchNormalization())

model.add(Conv2D(256, (5,5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
# Local Response normalization for Original Alexnet
model.add(BatchNormalization())

model.add(Conv2D(384, (3,3), activation='relu', padding='same'))
model.add(Conv2D(384, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
# Local Response normalization for Original Alexnet
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

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

y_true_labels = train_generator.class_indices

print(y_true_labels)

validation_generator = val_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


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
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator, steps=nb_samples)

print(predict)

target_dir = './models/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
model.save('./models/modelalexe.h5')
model.save_weights('./models/weightsalexe.h5')


results = hist.history.items()

print(results)

with open("results.txt", "w") as text_file:
    print(results, file=text_file)
