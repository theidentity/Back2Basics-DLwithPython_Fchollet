from keras.datasets import boston_housing
from keras.layers import Input, Dense
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.models import load_model, save_model
from keras.callbacks import EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras import losses
from keras import metrics
from keras import regularizers

import numpy as np
from helpers import remove
from helpers import normalize


class Boston_Regression(object):
    """docstring for Boston_Regression"""

    def __init__(self):
        self.inp_features = 13
        self.output_features = 1


        self.batch_size = 50
        self.train_samples = 404
        self.test_samples = 102
        self.steps_per_epoch = self.train_samples // self.batch_size + 1
        self.validation_steps = self.test_samples // self.batch_size + 1

        self.num_folds = 4

        self.name = ''.join(['boston', '_basic'])
        self.best_model_path = ''.join(
            ['models/', self.name, '_best', '.hdf5'])
        self.last_model_path = ''.join(
            ['models/', self.name, '_last', '.hdf5'])
        self.log_path = ''.join(['logs/', self.name, '_logs', '.csv'])
        self.graph_path = ''.join(['logs/', self.name, '_graph', '.png'])
        self.abs_error_path = ''.join(['logs/', self.name, '_abs_error', '.png'])

        self.add_dropout = False
        self.nodes_per_layer = [32]
        self.l2_reg_value = 0
        # self.l2_reg_value = 1e-3

        self.seed = 42

    def get_data(self):
        (X_train, y_train), (X_test, y_test) = boston_housing.load_data()
        X = np.concatenate([X_train,X_test])
        y = np.concatenate([y_train,y_test])

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / std

        from helpers import split_data
        gen = split_data(X,y,num_folds=self.num_folds)

        (X_train, y_train), (X_test, y_test) = next(gen)
        return X_train, X_test, y_train, y_test

    def get_model(self, dropout=False):

        input_seq = Input(shape=(self.inp_features,))
        x = input_seq

        for num_nodes in self.nodes_per_layer:
            if self.l2_reg_value > 0:
                x = Dense(num_nodes, activation='relu', kernel_regularizer=regularizers.l2(
                    self.l2_reg_value))(x)
            else:
                x = Dense(num_nodes, activation='relu')(x)
            if self.add_dropout:
                x = Dropout(0.5)(x)

        prediction = Dense(self.output_features)(x)
        model = Model(inputs=[input_seq], outputs=[prediction])
        return model

    def build_model(self, lr=1e-2):

        model = self.get_model()
        opt = RMSprop(lr)
        model.compile(optimizer=opt,
                      loss=losses.mse,
                      metrics=['accuracy', 'mse', 'mae']
                      )
        return model

    def get_callbacks(self):
        checkpoint = ModelCheckpoint(self.best_model_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=1e-6, patience=10, verbose=1, mode='auto', baseline=None)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4,
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

    def load_model(self, req_model='best'):
        if req_model == 'last':
            model = load_model(self.last_model_path)
        else:
            model = load_model(self.best_model_path)
        return model

    def continue_train(self, lr=1e-4, epochs=1, resume_model='last'):

        model = self.load_model(resume_model)
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

        y_pred = model.predict(x=X_test,
                                    batch_size=self.batch_size,
                                    verbose=1,
                                    steps=None
                                    )

        y_true = y_test

        y_pred = y_pred.reshape(-1,1)
        y_true = y_true.reshape(-1,1)

        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_true, y_pred)
        print('MAE : ',mae)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)
        print('MSE : ',mse)

        from helpers import plot_model_history
        addn_info = 'MSE : ' + str(round(mse, 3))
        plot_model_history(self.log_path, self.graph_path, addn_info=addn_info)

        from helpers import plot_regression
        plot_regression(y_true, y_pred, self.abs_error_path, addn_info=addn_info)


if __name__ == '__main__':

    clf = Boston_Regression()
    clf.get_data()
    # clf.train(lr=1e-2, epochs=500)
    # clf.continue_train(lr=1e-4, epochs=500, resume_model='last')
    # clf.evaluate()
    # del (clf)
