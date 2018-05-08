from bin.trainer import Trainer
from configs.drconfig import DR_CONFIG


from preprocessing.localaverage import localAverage
from test.drprediction import predictdr


"""
Parameters
"""
PARAMETERS = {
    'img_width' : 500,
    'img_height' : 500,
    'batch_size' : 16,
    'samples_per_epoch' : 1000,
    'validation_steps' : 10,
    'classes_num' : 5,
    'lr' : 0.0001,
    'epochs': 1

}


def main():
    # sorting datasets to classdirs and split train/validation
    # sort(DR_CONFIG)

    # preprocess images: resize and filter
    # hefilter(DR_CONFIG)
    # localAverage(DR_CONFIG)


    trainer = Trainer(DR_CONFIG, PARAMETERS)
    trainer.start()


    #predictdr(DR_CONFIG, PARAMETERS['img_width'], PARAMETERS['img_height'])


if __name__ == "__main__":
    main()
