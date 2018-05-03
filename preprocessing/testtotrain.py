import pandas as pd
import shutil

lines = pd.read_csv("./datasets/trainLabels.csv")


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
    file = i[0] + "-resized"
    level = i[1]

    filenr = file.split('_')[0]

    if int(filenr) > 40000:
        filelevel = getlevel(level)

        shutil.move("/home/determinants/automl/datasets/train/" + str(filelevel) + "/" + file + ".jpeg",
                    "/home/determinants/automl/datasets/validation/" + str(filelevel) + "/" + file + ".jpeg")
        # shutil.move("/home/determinants/automl/datasets/validation/" + str(filelevel) + "/" + file + ".jpeg"
        #            , "/home/determinants/automl/datasets/train/" + str(filelevel) + "/" + file + ".jpeg")
