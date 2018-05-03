import os

import pandas as pd
import shutil
from PIL import Image

width = 500
height = 500

lines = pd.read_csv("/home/reserchr/PycharmProjects/automl/trainLabels.csv")


def getlevel(levelnr):
    switcher = {
        0: "nodr",
        1: "mild",
        2: "moderate",
        3: "severe",
        4: "proliferativedr"
    }

    return switcher.get(levelnr, "invalid")


for i in lines.values:
    file = i[0]
    level = i[1]

    filenr = file.split('_')[0]

    if int(filenr) >= 26243:
        extension = ".jpeg"

        filelevel = getlevel(level)

        loaddir = "/home/reserchr/PycharmProjects/automl/train/" + str(filelevel)
        savedir = "/home/reserchr/PycharmProjects/automl/train/" + str(filelevel)

        # Open the image file.
        img = Image.open(os.path.join(loaddir, file + extension))

        # Resize it.
        img = img.resize((width, height), Image.ANTIALIAS)

        # Save it back to disk.
        img.save(os.path.join(savedir, file + '-train' + extension))

        print(filenr + " / ~45000 " + "images train")
