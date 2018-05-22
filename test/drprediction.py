import csv
import os

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt


def load_image(img_path, width, height, show=False):
    img = image.load_img(img_path, target_size=(width, height))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


def createSubmissionFile(filepath):
    with open(filepath, 'w') as csvfile:
        #filewriter = csv.writer(csvfile, delimiter=',',
         #                       quotechar='|', quoting=csv.QUOTE_MINIMAL)

        csvfile.write('image' + "," + 'level' + "\n")
        csvfile.close()


def appendToFile(filepath, file, prediction):
    with open(filepath, 'a') as csvfile:
        #filewriter = csv.writer(csvfile, delimiter=',',
        #                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvfile.write(file + "," + prediction + "\n")
        csvfile.close()

def predictdr(config, width, height):
    model = load_model("/home/determinants/automl/reports/custom_32_15_ep_85_val_loss_0.69_val_acc_0.81.h5")

    testdir = config['loaddir'] + "/la/test/"

    files = os.listdir(testdir)

    counter = 0
    filecount = len(files)

    submissionFilePath = "/home/determinants/automl/datasets/diabetic-retinopathy-detection/customsubmission069.csv"
    createSubmissionFile(submissionFilePath)

    for file in files:
        new_image = load_image(testdir + file, width, height)
        pred = model.predict(new_image)
        classes = pred.argmax(axis=-1)
        appendToFile(submissionFilePath, file.replace(".jpeg", ""), str(classes[0]))

        counter += 1
        print(str(counter) + " / " + str(filecount))
