from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.layers import Dropout, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.models import load_model, save_model
from keras.callbacks import EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from helpers import remove


class Img_Classifier(object):
    """docstring for Img_Classifier"""

    def __init__(self, sz):
        self.img_rows = sz
        self.img_cols = sz
        self.img_ch = 3

        self.batch_size = 128
        self.train_samples = 10000
        self.valid_samples = 4000
        self.steps_per_epoch = self.train_samples // self.batch_size + 1
        self.validation_steps = self.valid_samples // self.batch_size + 1

        self.train_imgs_path = 'data/catsdogs/train_imgs/'
        self.valid_imgs_path = 'data/catsdogs/valid_imgs/'

        self.name = ''.join(['cats_dogs', '_cnn'])
        self.best_model_path = ''.join(
            ['models/', self.name, '_best', '.hdf5'])
        self.last_model_path = ''.join(
            ['models/', self.name, '_last', '.hdf5'])
        self.log_path = ''.join(['logs/', self.name, '_logs', '.csv'])
        self.graph_path = ''.join(['logs/', self.name, '_graph', '.png'])

        self.global_avg_pool = True
        self.use_dropout = False
        self.threshold = 0.5

    def get_train_generator(self):

        img_gen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0.0,
            width_shift_range=0.10,
            height_shift_range=0.10,
            brightness_range=None,
            shear_range=0.0,
            zoom_range=0.20,
            channel_shift_range=0.0,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=1 / 255.0,
            preprocessing_function=None
        )

        img_gen = img_gen.flow_from_directory(
            directory=self.train_imgs_path,
            target_size=(self.img_rows, self.img_cols),
            color_mode='rgb',
            class_mode='binary',
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            interpolation='nearest'
        )

        return img_gen

    def get_valid_generator(self):

        img_gen = ImageDataGenerator(
            rescale=1 / 255.0,
            preprocessing_function=None
        )

        img_gen = img_gen.flow_from_directory(
            directory=self.valid_imgs_path,
            target_size=(self.img_rows, self.img_cols),
            color_mode='rgb',
            class_mode='binary',
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            interpolation='nearest'
        )

        return img_gen

    def get_model(self, dropout=False):

        input_img = Input(shape=(self.img_rows, self.img_cols, self.img_ch))
        x = input_img

        x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(x)
        x = MaxPool2D((2, 2))(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
        x = MaxPool2D((2, 2))(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
        x = MaxPool2D((2, 2))(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)

        if self.global_avg_pool:
            x = GlobalAveragePooling2D()(x)
        else:
            x = Flatten()(x)

        if self.use_dropout:
            x = Dropout(0.5)

        x = Dense(512, activation='relu')(x)
        prediction = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[input_img], outputs=[prediction])
        model.summary()
        return model

    def build_model(self, lr=1e-2):

        model = self.get_model()
        opt = RMSprop(lr)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_accuracy']
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

        train_generator = self.get_train_generator()
        valid_generator = self.get_valid_generator()

        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            verbose=1,
            callbacks=self.get_callbacks(),
            validation_data=valid_generator,
            validation_steps=self.validation_steps,
            class_weight=None,
            shuffle=True,
        )
        save_model(model, self.last_model_path)

    def continue_train(self, lr=1e-4, epochs=1, resume_model='last'):

        if resume_model == 'last':
            model = load_model(self.last_model_path)
        else:
            model = load_model(self.best_model_path)
        model.summary()

        train_generator = self.get_train_generator()
        valid_generator = self.get_valid_generator()

        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            verbose=1,
            callbacks=self.get_callbacks(),
            validation_data=valid_generator,
            validation_steps=self.validation_steps,
            class_weight=None,
            shuffle=True,
        )
        save_model(model, self.last_model_path)

    def evaluate(self):

        model = load_model(self.best_model_path)
        valid_generator = self.get_valid_generator()

        steps = self.validation_steps

        y_pred_prob = np.empty(shape=(0, 1), dtype=np.int32)
        y_true = np.empty(shape=(0, 1), dtype=np.int32)

        for i in range(steps):
            print(steps - i)
            imgs, labels = next(valid_generator)
            pred = model.predict_on_batch(imgs)
            labels = labels.reshape(-1, 1)
            y_pred_prob = np.vstack([y_pred_prob, pred])
            y_true = np.vstack([y_true, labels])

            y_pred = y_pred_prob > self.threshold
            y_pred = y_pred * 1

        from helpers import get_clf_report
        cm, report, acc, prec, recall, auc = get_clf_report(y_true, y_pred,y_pred_prob)

        from helpers import plot_model_history
        addn_info = 'Acc : ' + str(round(acc, 3))
        plot_model_history(self.log_path, self.graph_path, addn_info=addn_info)

    def debug(self):

        valid_generator = self.get_valid_generator()
        steps = self.validation_steps

        y_true = np.empty(shape=(0, 1), dtype=np.int32)

        for i in range(steps):
            imgs, labels = next(valid_generator)
            labels = labels.reshape(-1, 1)
            y_true = np.vstack([y_true, labels])
            print(y_true.shape)

        print(np.unique(y_true, return_counts=True))

if __name__ == '__main__':

    clf = Img_Classifier(sz=128)
    # clf.get_model()
    # clf.debug()

    # clf.train(lr=1e-3, epochs=20)
    # clf.continue_train(lr=1e-4, epochs=100, resume_model='last')
    clf.evaluate()
    del (clf)
