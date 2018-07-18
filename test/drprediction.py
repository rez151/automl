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
    model = load_model("/home/determinants/automl/models/inceptionv3/inceptionv3_ep_1_val_loss_1.31_val_acc_0.39.h5")

    testdir = config['loaddir'] + "/preprocess/512/test/"

    files = os.listdir(testdir)

    counter = 0
    filecount = len(files)

    submissionFilePath = "/home/determinants/automl/datasets/diabetic-retinopathy-detection/inception.csv"
    createSubmissionFile(submissionFilePath)

    for file in files:
        print(str(counter) + " / " + str(filecount) + ": " + str(file))
        new_image = load_image(testdir + file, width, height)
        pred = model.predict(new_image)
        classes = pred.argmax(axis=-1)
        appendToFile(submissionFilePath, file.replace(".jpeg", ""), str(classes[0]))

        counter += 1


def predict_all_models(config, width, height):
    loaddir = "/home/determinants/automl/reports/"
    directory = os.fsencode(loaddir)

    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        print(loaddir + filename)

        model = load_model(loaddir + filename)

        testdir = config['loaddir'] + "/la/test500/"

        files = os.listdir(testdir)

        counter = 0
        filecount = len(files)

        submissionFilePath = "/home/determinants/automl/datasets/diabetic-retinopathy-detection/" + filename +".csv"

        if submissionFilePath != "/home/determinants/automl/datasets/diabetic-retinopathy-detection/custom_32_10_ep_82_val_loss_1.60_val_acc_0.22.h5.csv":

            createSubmissionFile(submissionFilePath)

            for file in files:
                new_image = load_image(testdir + file, width, height)
                pred = model.predict(new_image)
                classes = pred.argmax(axis=-1)
                appendToFile(submissionFilePath, file.replace(".jpeg", ""), str(classes[0]))

                counter += 1
                print(str(counter) + " / " + str(filecount))
