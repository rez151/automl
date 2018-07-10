import sys
sys.path.insert(0,'/home/determinants/automl')

from mainscripts.trainer import Trainer
from configs.drconfig import DR_CONFIG

from preprocessing.localaverage import localAverage
from test.drprediction import predictdr, predict_all_models

"""
Parameters
"""
PARAMETERS = {
    'num_gpus' : 2,
    'img_width' : 512,
    'img_height' : 512,
    'batch_size' : 32,
    'samples_per_epoch' : 5000,
    'validation_steps' : 3,
    'classes_num' : 5,
    'lr' : 0.05,
    'epochs': 1000,
    'patience' : 50

}


def main():
    # sorting datasets to classdirs and split train/validation
    # sort(DR_CONFIG)

    # preprocess images: resize and filter
    # hefilter(DR_CONFIG)
    # localAverage(DR_CONFIG)


    trainer = Trainer(DR_CONFIG, PARAMETERS)
    trainer.start()

    #from preprocessing.resizer import resize_test
    #resize_test(512,512)

    #predictdr(DR_CONFIG, PARAMETERS['img_width'], PARAMETERS['img_height'])
    #predict_all_models(DR_CONFIG, PARAMETERS['img_width'], PARAMETERS['img_height'])


if __name__ == "__main__":
    main()
