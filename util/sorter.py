import pandas as pd
import shutil

lines = pd.read_csv("/home/reserchr/PycharmProjects/automl/trainLabels.csv")

for i in lines.values:
    file = i[0]
    level = i[1]

    shutil.move("/home/reserchr/PycharmProjects/automl/train/" + file + ".jpeg",
                "/home/reserchr/PycharmProjects/automl/train/" + str(level) + "/" + file + ".jpeg")
