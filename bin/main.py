from configs.drconfig import DR_CONFIG
from preprocessing.filter import hefilter
from preprocessing.sorter import sort, revert


# image preprocessing (histogram equalization, resizing, sorting)

# train models (safe after each epoch)


def main():

    #sorting datasets to classdirs and split train/validation
    #sort(DR_CONFIG)

    #preprocess images: resize and filter
    hefilter(DR_CONFIG)







if __name__ == "__main__":
    main()
