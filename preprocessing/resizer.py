import os

import pandas as pd
import shutil
from PIL import Image

width = 512
height = 512

lines = pd.read_csv("/home/determinants/automl/datasets/diabetic-retinopathy-detection/trainLabels.csv")


def getlevel(levelnr):
    switcher = {
        0: "0_nodr",
        1: "1_mild",
        2: "2_moderate",
        3: "3_severe",
        4: "4_proliferativedr"
    }

    return switcher.get(levelnr, "invalid")


def resize_train(height, width):
    for i in lines.values:
        file = i[0]
        level = i[1]

        filenr = file.split('_')[0]

        extension = ".jpeg"

        filelevel = getlevel(level)

        loaddir = "/home/determinants/automl/datasets/diabetic-retinopathy-detection/la/train/" + str(filelevel)
        savedir = "/home/determinants/automl/datasets/diabetic-retinopathy-detection/la/500/" + str(filelevel)

        try:

            # Open the image file.
            img = Image.open(os.path.join(loaddir, file + extension))

            # Resize it.
            img = img.resize((width, height), Image.ANTIALIAS)

            # Save it back to disk.
            img.save(os.path.join(savedir, file + extension))
        except:
            print("file " + str(i) + " not found")

        print(filenr + " / " + str(lines.size) + "images train")


def resize_test(height, width):
    loaddirtest = "/home/determinants/automl/datasets/diabetic-retinopathy-detection/la/test"
    savedirtest = "/home/determinants/automl/datasets/diabetic-retinopathy-detection/la/test500"
    files = os.listdir(loaddirtest)

    testcount = len(files)
    counter = 0
    for f in files:
        if not os.path.exists(savedirtest + "/" + f):
            # Open the image file.
            img = Image.open(os.path.join(loaddirtest, f))

            # Resize it.
            img = img.resize((width, height), Image.ANTIALIAS)

            # Save it back to disk.
            img.save(os.path.join(savedirtest, f))

        counter += 1
        if (counter % 100) == 0:
            print(str(counter) + " / " + str(testcount))
