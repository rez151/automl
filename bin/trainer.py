import os

from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from models.architectures import load_alexnet, load_inception_v3


class Trainer(object):

    def __init__(self, config, parameters):
        # load parameters for diabetic retinopathy
        self.__name = config['name']
        self.__loaddir = config['loaddir']
        self.__labelfile = config['labelfile']
        self.__fileextension = config['fileextension']
        self.__classes = config['classes']
        self.__class_weights = config['classweights']
        self.__validation_split = config['validationsplit']

        # load parameters for training
        self.__width = parameters['img_width']
        self.__height = parameters['img_height']
        self.__batch_size = parameters['batch_size']
        self.__samples_per_epoch = parameters['samples_per_epoch']
        self.__validation_steps = parameters['validation_steps']
        self.__classes_num = parameters['classes_num']
        self.__lr = parameters['lr']
        self.__epochs = parameters['epochs']

        # get imagedatagenerators
        self.__train_generator, self.__validation_generator = self.get_generators()

        # load
        self.__models = {
            'inceptionv3': load_inception_v3(self.__width, self.__height, self.__classes_num, self.__lr),
            'alexnet': load_alexnet(self.__width, self.__height, self.__classes_num, self.__lr)
        }

    def get_generators(self):
        train_data_path = self.__loaddir + "/la/train"
        validation_data_path = self.__loaddir + "/la/validation"

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.1
        )

        val_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.1
        )

        train_generator = train_datagen.flow_from_directory(
            train_data_path,
            target_size=(self.__height, self.__width),
            batch_size=self.__batch_size,
            class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(
            validation_data_path,
            target_size=(self.__height, self.__width),
            batch_size=self.__batch_size,
            class_mode='categorical')

        return train_generator, validation_generator

    def start(self):

        model = load_alexnet(self.__width, self.__height, self.__classes_num, self.__lr)
        self.train('alexnet', model)

        clear_session()
        model = load_inception_v3(self.__width, self.__height, self.__classes_num, self.__lr)
        self.train('inception', model)

        #for modelname in self.__models:
        #    print("train with: " + modelname)
#
        #    clear_session()
#
        #    self.train(modelname, self.__models[modelname])

        # trainWithAlexnet()
        # trainWithXception()
        # trainWithVGG16()
        # trainWithVGG19()
        # trainWithResNet50()
        # trainWithInceptionV3()
        # trainWithInceptionResNetV2()
        # trainWithMobileNet()
        # trainWithDenseNet()
        # trainWithNASNet()

        return 0

    def train(self, modelname, model):
        checkpoint = ModelCheckpoint(
            '/home/determinants/automl/reports/' + modelname + '_' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

        model.fit_generator(
            generator=self.__train_generator,
            samples_per_epoch=self.__samples_per_epoch,
            epochs=self.__epochs,
            steps_per_epoch=31,
            validation_data=self.__validation_generator,
            validation_steps=self.__validation_steps,
            class_weight=self.__class_weights,
            callbacks=[checkpoint]
        )

            # model.
    #
    # target_dir = '/home/determinants/automl/models/trained'
    # if not os.path.exists(target_dir):
    #    os.mkdir(target_dir)
    # model.save('/home/determinants/automl/models/trained/modelalexe.h5')
    # model.save_weights('/home/determinants/automl/models/trained/weightsalexe.h5')
