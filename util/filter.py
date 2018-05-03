import cv2 as cv2
import pandas as pd

lines = pd.read_csv("/home/determinants/automl/datasets/trainLabels.csv")


def getlevel(levelnr):
    switcher = {
        0: "nodr/",
        1: "mild/",
        2: "moderate/",
        3: "severe/",
        4: "proliferativedr/"
    }

    return switcher.get(levelnr, "invalid")


for i in lines.values:
    file = i[0]
    level = i[1]

    filenr = file.split('_')[0]
    filelevel = getlevel(level)
    extension = ".jpeg"

    loaddir = "/home/determinants/automl/datasets/train/" + str(filelevel)
    savedir = "/home/determinants/automl/datasets/filtered/train/" + str(filelevel)

    if int(filenr) > 40000:
        loaddir = "/home/determinants/automl/datasets/validation/" + str(filelevel)
        savedir = "/home/determinants/automl/datasets/filtered/validation/" + str(filelevel)

    # Open the image file.
    img = cv2.imread(loaddir + file + '-resized' + extension, 0)

    # Apply filter
    equ = cv2.equalizeHist(img)

    # Save it back to disk.
    cv2.imwrite(savedir + file + '-he' + extension, equ)

    print(filenr + " / 35127 images filtered")

exit()
