import os

import cv2
import numpy
import pandas as pd

from util.directories import createDirs


def scaleRadius(img, scale):
    try:
        x = img[img.shape[0] // 2, :, :].sum(1)

        r = (x > x.mean() / 10).sum() / 2

        s = scale * 1.0 / r
        return cv2.resize(img, (0, 0), fx=s, fy=s)

    except:
        print("r is 0")


scale = 300


def localAverage(config):
    labelfile = config['labelfile']
    loaddir = config['loaddir']
    classes = config['classes']
    validationsplit = config['validationsplit']
    extension = config['fileextension']

    lines = pd.read_csv(labelfile)

    imgcount = lines.size / 2
    validcount = imgcount * validationsplit

    savedir = loaddir + "/la"
    createDirs(savedir, classes)

    counter = 0

    for i in lines.values:
        file = i[0]
        level = i[1]

        if counter < imgcount - validcount:
            if not os.path.exists(savedir + "/train/" + classes[level] + "/" + file + extension):
                a = cv2.imread(loaddir + "/train/" + classes[level] + "/" + file + extension)
                # scale img to a given radius
                a = scaleRadius(a, scale)
                # subtract local mean color
                a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
                # remove outer 10%
                b = numpy.zeros(a.shape)
                cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
                a = a * b + 128 * (1 - b)
                cv2.imwrite(savedir + "/train/" + classes[level] + "/" + file + extension, a)

        else:
            if not os.path.exists(savedir + "/validation/" + classes[level] + "/" + file + extension):
                a = cv2.imread(loaddir + "/validation/" + classes[level] + "/" + file + extension)
                # scale img to a given radius
                a = scaleRadius(a, scale)
                # subtract local mean color
                a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
                # remove outer 10%
                b = numpy.zeros(a.shape)
                cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
                a = a * b + 128 * (1 - b)
                cv2.imwrite(savedir + "/validation/" + classes[level] + "/" + file + extension, a)

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
            a = cv2.imread(loaddir + "/test/" + f)
            # scale img to a given radius
            a = scaleRadius(a, scale)
            # subtract local mean color
            a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
            # remove outer 10%
            b = numpy.zeros(a.shape)
            cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
            a = a * b + 128 * (1 - b)
            cv2.imwrite(savedir + "/test/" + f, a)

        counter += 1
        if (counter % 100) == 0:
            print(str(counter) + " / " + str(testcount))

    print("filtering done")
