import os

from keras.preprocessing.image import ImageDataGenerator

from models.alexnet import getAlexnet


def trainWithAlexnet(loaddir, width, height, classes_num, epochs, lr, batch_size, samples_per_epoch, validation_steps,
                     weights):
    train_data_path = loaddir + "/he/train"
    validation_data_path = loaddir + "/he/validation"

    model = getAlexnet(width, height, classes_num, lr)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        validation_data_path,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')

    model.fit_generator(
        generator=train_generator,
        samples_per_epoch=samples_per_epoch,
        epochs=epochs,
        steps_per_epoch=31,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=weights
    )

    target_dir = '/home/determinants/automl/models/trained'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model.save('/home/determinants/automl/models/trained/modelalexe.h5')
    model.save_weights('/home/determinants/automl/models/trained/weightsalexe.h5')
