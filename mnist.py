from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.models import load_model, save_model
from keras.callbacks import EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

import numpy as np
from helpers import remove


class MNIST_Classifier(object):
    """docstring for MNIST_Classifier"""

    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.num_classes = 10

        self.batch_size = 1000
        self.train_samples = 60000
        self.test_samples = 10000
        self.steps_per_epoch = self.train_samples // self.batch_size + 1
        self.validation_steps = self.test_samples // self.batch_size + 1

        self.name = ''.join(['mnist', '_norm'])
        self.best_model_path = ''.join(
            ['models/', self.name, '_best', '.hdf5'])
        self.last_model_path = ''.join(
            ['models/', self.name, '_last', '.hdf5'])
        self.log_path = ''.join(['logs/', self.name, '_logs', '.csv'])
        self.graph_path = ''.join(['logs/', self.name, '_graph', '.png'])

        self.normalize_data = True
        self.add_dropout = False
        self.nodes_per_layer = [64, 64, 64]

    def get_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        if self.normalize_data:
            mean = np.mean(X_train)
            std = np.std(X_train)
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
        else:
            X_train = x_train / 255.0
            X_test = x_test / 255.0

        X_train = X_train.reshape(-1, self.img_rows * self.img_cols)
        X_test = X_test.reshape(-1, self.img_rows * self.img_cols)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return X_train, X_test, y_train, y_test

    def get_model(self, dropout=False):

        input_nodes = [8, 8, 8]

        input_img = Input(shape=(self.img_rows * self.img_cols,))
        x = input_img

        for num_nodes in self.nodes_per_layer:
            x = Dense(num_nodes, activation='relu')(x)
            if self.add_dropout:
                x = Dropout(0.5)(x)

        prediction = Dense(self.num_classes, activation='sigmoid')(x)
        model = Model(inputs=[input_img], outputs=[prediction])
        return model

    def build_model(self, lr=1e-2):

        model = self.get_model()
        opt = RMSprop(lr)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_accuracy']
                      )
        return model

    def get_callbacks(self):
        checkpoint = ModelCheckpoint(self.best_model_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=1e-6, patience=5, verbose=1, mode='auto', baseline=None)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                      verbose=1, mode='auto', min_delta=1e-6, cooldown=0, min_lr=1e-7)
        csv_logger = CSVLogger(self.log_path, append=True)
        return [checkpoint, csv_logger,
                reduce_lr, early_stopping
                ]

    def train(self, lr=1e-2, epochs=1):

        remove(self.log_path)
        model = self.build_model(lr)
        model.summary()

        X_train, X_test, y_train, y_test = self.get_data()
        model.fit(x=X_train,
                  y=y_train,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=self.get_callbacks(),
                  validation_data=(X_test, y_test),
                  shuffle=True,
                  initial_epoch=0,
                  steps_per_epoch=None, validation_steps=None)

        save_model(model, self.last_model_path)

    def continue_train(self, lr=1e-4, epochs=1, resume_model='last'):

        if resume_model == 'last':
            model = load_model(self.last_model_path)
        else:
            model = load_model(self.best_model_path)
        model.summary()

        X_train, X_test, y_train, y_test = self.get_data()
        model.fit(x=X_train,
                  y=y_train,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=self.get_callbacks(),
                  validation_data=(X_test, y_test),
                  shuffle=True,
                  # initial_epoch=start_epoch,
                  steps_per_epoch=None, validation_steps=None)
        save_model(model, self.last_model_path)

    def evaluate(self):

        X_train, X_test, y_train, y_test = self.get_data()
        model = load_model(self.best_model_path)

        y_pred_prob = model.predict(x=X_test,
                                    batch_size=self.batch_size,
                                    verbose=1,
                                    steps=None
                                    )

        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred_prob, axis=1)

        from helpers import get_clf_report
        cm, report, acc = get_clf_report(y_true, y_pred)

        from helpers import plot_model_history
        addn_info = 'Acc : ' + str(round(acc, 3))
        plot_model_history(self.log_path, self.graph_path, addn_info=addn_info)


if __name__ == '__main__':

    clf = MNIST_Classifier()
    clf.train(lr=1e-3, epochs=100)
    clf.continue_train(lr=1e-4, epochs=100, resume_model='last')
    clf.evaluate()
    del (clf)
