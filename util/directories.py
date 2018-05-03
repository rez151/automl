import os


def createDirs(loaddir, classes):

    if not os.path.exists(loaddir):
        os.makedirs(loaddir)

    for c in classes:
        if not os.path.exists(loaddir + "/train/" + c):
            os.makedirs(loaddir + "/train/" + c)

        if not os.path.exists(loaddir + "/validation/" + c):
            os.makedirs(loaddir + "/validation/" + c)
