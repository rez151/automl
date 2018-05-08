from bin.trainer import trainWithAlexnet, trainDataset
from configs.drconfig import DR_CONFIG

# image preprocessing (histogram equalization, resizing, sorting)

# train models (safe after each epoch)
from preprocessing.localaverage import localAverage
from test.drprediction import predictdr

"""
Parameters
"""
img_width, img_height = 500, 500
batch_size = 48
samples_per_epoch = 1000
validation_steps = 10
classes_num = 5
lr = 0.0001
epochs = 200


def main():
    # sorting datasets to classdirs and split train/validation
    # sort(DR_CONFIG)

    # preprocess images: resize and filter
    # hefilter(DR_CONFIG)
    localAverage(DR_CONFIG)

    trainDataset(DR_CONFIG)

    trainWithAlexnet(DR_CONFIG['loaddir'], img_width, img_height, classes_num, epochs, lr, batch_size,
                     samples_per_epoch, validation_steps, DR_CONFIG['classweights'])

    predictdr(DR_CONFIG, img_width, img_height)


if __name__ == "__main__":
    main()
