import shutil

import pandas as pd

from util.directories import createDirs


def sort(config):
    labelfile = config['labelfile']
    loaddir = config['loaddir']
    classes = config['classes']
    validationsplit = config['validationsplit']
    extension = config['fileextension']

    lines = pd.read_csv(labelfile)

    imgcount = lines.size / 2
    validcount = imgcount * validationsplit

    createDirs(loaddir, classes)

    counter = 0

    for i in lines.values:
        file = i[0]
        level = i[1]

        if counter < imgcount - validcount:
            shutil.move(loaddir + "/train/" + file + extension,
                        loaddir + "/train/" + classes[level] + "/" + file + extension)
        else:
            shutil.move(loaddir + "/train/" + file + extension,
                        loaddir + "/validation/" + classes[level] + "/" + file + extension)
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