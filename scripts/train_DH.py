import os.path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping

from classifier.datasets import BRIS, DH
from classifier.models import IMAGE_WIDTH, DocModel

ROOT = os.path.join(os.path.dirname(__file__), '..')
FOLDER_FEATURES = os.path.join(ROOT, r'data/features')
FILENAME_X = os.path.join(FOLDER_FEATURES, f'x_DH_BRIS_{IMAGE_WIDTH}.npy')
FILENAME_Y = os.path.join(FOLDER_FEATURES, f'y_DH_BRIS_{IMAGE_WIDTH}.npy')


def gen_y_from_x(x: np.ndarray, label: int):
    return label * np.ones((x.shape[0],), dtype=np.int)


def get_data(b_scratch: bool = False) -> (np.ndarray, np.ndarray):
    """
    Training data for BRIS vs DH use-case

    Args:
        b_scratch:

    Returns:

    """
    if b_scratch or (not os.path.exists(FILENAME_X)) or (not os.path.exists(FILENAME_Y)):
        x_BRIS = BRIS()
        y_BRIS = gen_y_from_x(x_BRIS, 0)

        x_DH = DH()
        y_DH = gen_y_from_x(x_DH, 1)

        x = np.concatenate([x_BRIS, x_DH], axis=0)
        y = np.concatenate([y_BRIS, y_DH], axis=0)

        np.save(FILENAME_X, x)
        np.save(FILENAME_Y, y)

    else:
        x = np.load(FILENAME_X)
        y = np.load(FILENAME_Y)

    return x, y


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 batch_size=32,
                 shuffle=True):
        'Initialization'
        self.x = x
        self.y = y

        assert len(x) == len(y), f'{x.shape}, {y.shape}'

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.x[indexes]
        y = self.y[indexes]

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indices)


def main():
    # dataset = image_dataset.image_dataset_from_directory(FOLDER_BOG,
    #                                                      image_size=(IMAGE_WIDTH, IMAGE_WIDTH),
    #                                                      label_mode=None)

    x, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=123)

    net = DocModel()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    if 0:
        # Does not work, out of memory

        hist = net.fit_fast(X_train, y_train,
                            epochs=1000,
                            validation_data=(X_test, y_test),
                            callbacks=es)

    else:
        # Use data generators

        if 0:
            training_generator = DataGenerator(X_train,
                                               y_train,
                                               )

            net.fit_generator(generator=training_generator,
                              # validation_data=validation_generator,
                              callbacks=es,
                              epochs=2
                              # use_multiprocessing=True,
                              # workers=6
                              )

        else:
            # Only train last layers
            f_train = net.feature(X_train)
            f_valid = net.feature(X_test)

            training_generator_f = DataGenerator(f_train,
                                                 y_train,
                                                 )

            valid_generator_f = DataGenerator(f_valid,
                                              y_test,
                                              )

            net._model_classifier.fit(training_generator_f,
                                      validation_data=valid_generator_f,
                                      epochs=200,
                                      callbacks=[es],
                                      )

    return


if __name__ == '__main__':
    main()
