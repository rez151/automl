import os

import keras
from ipython_genutils.py3compat import xrange
from keras import optimizers
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import *
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model, plot_model
from sklearn.utils import compute_class_weight

from mainscripts.savecallback import SaveCallback
from models.architectures import *


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
        self.__num_gpus = parameters['num_gpus']
        self.__width = parameters['img_width']
        self.__height = parameters['img_height']
        self.__batch_size = parameters['batch_size']
        self.__samples_per_epoch = parameters['samples_per_epoch']
        self.__validation_steps = parameters['validation_steps']
        self.__classes_num = parameters['classes_num']
        self.__lr = parameters['lr']
        self.__epochs = parameters['epochs']
        self.__patience = parameters['patience']

        self.counter = 0

        # get imagedatagenerators
        self.__train_generator, self.__validation_generator = self.get_generators()

        # instantiate optimizers
        self.__optimizers = {
            'sgd': SGD(),
            'rmsprop': RMSprop(),
            'adagrad': Adagrad(),
            'adadelta': Adadelta(),
            'adam': Adam(),
            'adamax': Adamax(),
            'nadam': Nadam()
        }

    def get_generators(self):


        train_data_path = self.__loaddir + "/equal/train"

        # files_per_class = []
        # for folder in sorted(os.listdir(train_data_path)):
        #    if not os.path.isfile(folder):
        #        files_per_class.append(len(os.listdir(train_data_path + '/' + folder)))
        # total_files = sum(files_per_class)
        # class_weights = {}
        # for i in xrange(len(files_per_class)):
        #    class_weights[i] = 1 - (float(files_per_class[i]) / total_files)
        # print(class_weights)

        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=360,
            validation_split=0.1
        )

        train_generator = datagen.flow_from_directory(
            train_data_path,
            target_size=(self.__height, self.__width),
            batch_size=self.__batch_size,
            class_mode='categorical',
            subset='training'
            #save_to_dir='/home/determinants/automl/datasets/diabetic-retinopathy-detection/savedir'
        )


        validation_data = datagen.flow_from_directory(
            train_data_path,
            target_size=(self.__height, self.__width),
            batch_size=self.__batch_size,
            class_mode='categorical',
            subset='validation'
        )


        return train_generator, validation_data


    def start(self):

        for optimizer in self.__optimizers:

            #for f in range(32, 512, 32):
            #    for l in range(5, 30, 5):
            #        model = load_customnet(self.__width, self.__height, self.__classes_num, l, f, 512)
            #        self.train('custom_' + str(f) + "_" + str(l), model, optimizer)
            #        self.counter += 1

            #model = load_customnet(self.__width, self.__height, self.__classes_num, 50, 256, 512, dense_units=2048)
            #self.train('custom_' + str(30) + "_" + str(256) + "_" + str(512), model, optimizer)

            #model = load_usuyama(self.__width, self.__height, self.__classes_num)
            #self.train('usuyama', model, optimizer)

            model = load_inception_v3(self.__width, self.__height, self.__classes_num)
            self.train('inceptionv3', model, optimizer)

            #model = load_m42(self.__width, self.__height, self.__classes_num)
            #self.train('m42', model, optimizer)
#
            #model = load_2nd_place(self.__width, self.__height, self.__classes_num)
            #self.train('second place', model, optimizer)
#
            #clear_session()
            #model = load_inception_v3(self.__width, self.__height, self.__classes_num)
            #self.train('inception', model, optimizer)

            '''
            model = load_alexnet(self.__width, self.__height, self.__classes_num)
            self.train('alexnet', model, optimizer)


            clear_session()
            model = load_xception(self.__width, self.__height, self.__classes_num)
            self.train('Xception', model, optimizer)

            clear_session()
            model = load_vgg16(self.__width, self.__height, self.__classes_num)
            self.train('VGG16', model, optimizer)

            clear_session()
            model = load_vgg19(self.__width, self.__height, self.__classes_num)
            self.train('VGG19', model, optimizer)

            clear_session()
            model = load_resnet50(self.__width, self.__height, self.__classes_num)
            self.train('ResNet50', model, optimizer)

            clear_session()
            model = load_inceptionresnet_v2(self.__width, self.__height, self.__classes_num)
            self.train('InceptionResNetV2', model, optimizer)

            clear_session()
            model = load_mobilenet(self.__width, self.__height, self.__classes_num)
            self.train('MobileNet', model, optimizer)

            clear_session()
            model = load_densenet121(self.__width, self.__height, self.__classes_num)
            self.train('DenseNet121', model, optimizer)

            clear_session()
            model = load_densenet169(self.__width, self.__height, self.__classes_num)
            self.train('DenseNet169', model, optimizer)

            clear_session()
            model = load_densenet201(self.__width, self.__height, self.__classes_num)
            self.train('DenseNet201', model, optimizer)
            '''
            return 0


    def train(self, modelname, model, optimizer):
        # checkpoint = ModelCheckpoint(
        #    '/home/determinants/automl/reports/' + modelname + '_' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        #    save_best_only=True)

        #plot_model(model, to_file='/home/determinants/automl/models/' + str(modelname) + '.png', show_shapes=True)

        print("train with: " + modelname)
        print("optimizer : " + optimizer)

        # DEFINE CALLBACKS
        save_callback = SaveCallback(model, modelname)
        early_stopping = EarlyStopping(patience=self.__patience)

        # USE MORE GPUS
        parallel_model = multi_gpu_model(model, gpus=2)

        parallel_model.compile(loss='categorical_crossentropy',
                               #optimizer=self.__optimizers[optimizer],
                               #optimizer=self.__optimizers['adam'],
                               optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                               metrics=['accuracy'])



        callbacks = [save_callback,
                     early_stopping,
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01,
                                       cooldown=0, min_lr=1e-6)]

        parallel_model.fit_generator(
            use_multiprocessing=True,
            generator=self.__train_generator,
            samples_per_epoch=self.__samples_per_epoch,
            epochs=self.__epochs,
            validation_data=self.__validation_generator,
            validation_steps=self.__validation_steps,
            #class_weight=self.__class_weights,
            callbacks=callbacks
        )

        model.save("/home/determinants/automl/reports/last" + str(self.counter) + ".h5")
