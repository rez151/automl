import os

import pandas as pd
import shutil


def createDirs(loaddir, classes):
    for c in classes:
        if not os.path.exists(loaddir + "/train/" + c):
            os.makedirs(loaddir + "/train/" + c)

        if not os.path.exists(loaddir + "/validation/" + c):
            os.makedirs(loaddir + "/validation/" + c)


def sort(config):
    labelfile = config['labelfile']
    loaddir = config['loaddir']
    classes = config['classes']
    validationsplit = config['validationsplit']

    lines = pd.read_csv(labelfile)

    imgcount = lines.size / 2
    validcount = imgcount * validationsplit

    createDirs(loaddir, classes)

    counter = 0

    for i in lines.values:
        file = i[0]
        level = i[1]

        if counter < imgcount - validcount:
            shutil.move(loaddir + "/train/" + file + ".jpeg",
                        loaddir + "/train/" + classes[level] + "/" + file + ".jpeg")
        else:
            shutil.move(loaddir + "/train/" + file + ".jpeg",
                        loaddir + "/validation/" + classes[level] + "/" + file + ".jpeg")
        counter += 1
        if (counter % 100) == 0:
            print(str(counter) + '/' + str(imgcount))

def revert(config):
    labelfile = config['labelfile']
    loaddir = config['loaddir']
    classes = config['classes']
    validationsplit = config['validationsplit']

    lines = pd.read_csv(labelfile)

    imgcount = lines.size / 2
    validcount = imgcount * validationsplit

    createDirs(loaddir, classes)

    counter = 0

    for i in lines.values:
        file = i[0]
        level = i[1]

        if counter < imgcount - validcount:
            shutil.move(loaddir + "/train/" + classes[level] + "/" + file + ".jpeg",
                        loaddir + "/train/" + file + ".jpeg")
        else:
            shutil.move(loaddir + "/validation/" + classes[level] + "/" + file + ".jpeg",
                        loaddir + "/train/" + file + ".jpeg")
        counter += 1