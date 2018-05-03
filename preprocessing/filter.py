import os

import cv2 as cv2
import pandas as pd

from util.directories import createDirs


def hefilter(config):
    labelfile = config['labelfile']
    loaddir = config['loaddir']
    classes = config['classes']
    validationsplit = config['validationsplit']
    extension = config['fileextension']

    lines = pd.read_csv(labelfile)

    imgcount = lines.size / 2
    validcount = imgcount * validationsplit

    savedir = loaddir + "/he"
    createDirs(savedir, classes)

    counter = 0

    for i in lines.values:
        file = i[0]
        level = i[1]

        if counter < imgcount - validcount:
            if not os.path.exists(savedir + "/train/" + classes[level] + "/" + file + extension):
                img = cv2.imread(loaddir + "/train/" + classes[level] + "/" + file + extension, 0)
                equ = cv2.equalizeHist(img)
                cv2.imwrite(savedir + "/train/" + classes[level] + "/" + file + extension, equ)
        else:
            if not os.path.exists(savedir + "/validation/" + classes[level] + "/" + file + extension):
                img = cv2.imread(loaddir + "/validation/" + classes[level] + "/" + file + extension, 0)
                equ = cv2.equalizeHist(img)
                cv2.imwrite(savedir + "/validation/" + classes[level] + "/" + file + extension, equ)

        counter += 1
        if (counter % 100) == 0:
            print(str(counter) + '/' + str(imgcount))

    print("converting testimages...")

    if not os.path.exists(savedir + "/test"):
        os.makedirs(savedir + "/test")

    files = os.listdir(loaddir + "/test/")

    testcount = len(files)
    counter = 0
    for f in files:
        if not os.path.exists(savedir + "/test/" + f):
            img = cv2.imread(loaddir + "/test/" + f, 0)
            equ = cv2.equalizeHist(img)
            cv2.imwrite(savedir + "/test/" + f, equ)

        counter += 1
        if (counter % 100) == 0:
            print(str(counter) + " / " + str(testcount))

    print("filtering done")
