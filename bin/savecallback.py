import keras


class SaveCallback(keras.callbacks.Callback):
    save_path = '/home/determinants/automl/reports/'

    def __init__(self, model, modelname):
        self.__best_loss = 100.0
        self.__best_acc = 0.0
        self.model_to_save = model
        self.__model_name = modelname

    def on_epoch_end(self, epoch, logs='val_loss'):

        if float(logs['val_loss']) < float(self.__best_loss):

            self.__best_loss = float(logs['val_loss'])

            self.model_to_save.save(self.save_path + self.__model_name + '_ep_' + str(epoch) +
                                    "_val_loss_" + str(
                '{:{width}.{prec}f}'.format(logs['val_loss'], width=4, prec=2)) +
                                    '_val_acc_' + str(
                '{:{width}.{prec}f}'.format(logs['val_acc'], width=4, prec=2) + '.h5'))
