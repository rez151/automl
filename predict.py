import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt



import numpy as np


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

"""
Parameters
"""
img_width, img_height = 224, 224
classes_num = 5

model = load_model("models/model20000e.h5")

#img_path = 'datasets/validation/proliferativedr/40178_left-resized.jpeg'
img_path = '/home/determinants/automl/datasets/train/mild/10234_left-resized.jpeg'

new_image = load_image(img_path)

pred = model.predict(new_image)
classes = pred.argmax(axis=-1)

print(pred)
print(classes)

img_path = '/home/determinants/automl/datasets/train/moderate/1032_right-resized.jpeg'

new_image = load_image(img_path)

pred = model.predict(new_image)
classes = pred.argmax(axis=-1)

print(pred)
print(classes)


img_path = '/home/determinants/automl/datasets/train/nodr/10009_left-resized.jpeg'

new_image = load_image(img_path)

pred = model.predict(new_image)
classes = pred.argmax(axis=-1)

print(pred)
print(classes)


img_path = '/home/determinants/automl/datasets/train/proliferativedr/11874_left-resized.jpeg'

new_image = load_image(img_path)

pred = model.predict(new_image)
classes = pred.argmax(axis=-1)

print(pred)
print(classes)


img_path = '/home/determinants/automl/datasets/train/severe/11529_left-resized.jpeg'

new_image = load_image(img_path)

pred = model.predict(new_image)
classes = pred.argmax(axis=-1)

print(pred)
print(classes)


