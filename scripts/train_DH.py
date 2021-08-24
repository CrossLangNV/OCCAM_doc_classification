import os.path

import numpy as np
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping

from classifier.datasets import BRIS, DH, Subset
from classifier.models import IMAGE_WIDTH, DocModel

ROOT = os.path.join(os.path.dirname(__file__), '..')
FOLDER_FEATURES = os.path.join(ROOT, r'data/features')
FILENAME_X = os.path.join(FOLDER_FEATURES, f'x_DH_BRIS_{IMAGE_WIDTH}_{{subset}}.npy')
FILENAME_Y = os.path.join(FOLDER_FEATURES, f'y_DH_BRIS_{IMAGE_WIDTH}_{{subset}}.npy')


def gen_y_from_x(x: np.ndarray, label: int):
    return label * np.ones((x.shape[0],), dtype=np.int)


def get_data(b_scratch: bool = False, subset:Subset=Subset.TRAIN) -> (np.ndarray, np.ndarray):
    """
    Training data for BRIS vs DH use-case

    Args:
        b_scratch:

    Returns:

    """

    filename_x = FILENAME_X.format(subset=subset.value)
    filename_y = FILENAME_Y.format(subset=subset.value)

    if b_scratch or (not os.path.exists(filename_x)) or (not os.path.exists(filename_x)):
        x_BRIS = BRIS(subset)
        y_BRIS = gen_y_from_x(x_BRIS, 0)

        x_DH = DH(subset)
        y_DH = gen_y_from_x(x_DH, 1)

        x = np.concatenate([x_BRIS, x_DH], axis=0)
        y = np.concatenate([y_BRIS, y_DH], axis=0)

        np.save(filename_x, x)
        np.save(filename_y, y)

    else:
        x = np.load(filename_x)
        y = np.load(filename_y)

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

    X_train, y_train = get_data(subset=Subset.TRAIN)
    X_valid, y_valid = get_data(subset=Subset.VALID)
    X_test, y_test = get_data(subset=Subset.TEST)

    net = DocModel()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    if 0:
        # Does not work, out of memory

        hist = net.fit_fast(X_train, y_train,
                            epochs=1000,
                            validation_data=(X_valid, y_valid),
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
            f_valid = net.feature(X_valid)

            training_generator_f = DataGenerator(f_train,
                                                 y_train,
                                                 )

            valid_generator_f = DataGenerator(f_valid,
                                              y_valid,
                                              )

            net._model_classifier.fit(training_generator_f,
                                      validation_data=valid_generator_f,
                                      epochs=200,
                                      callbacks=[es],
                                      )

    return


if __name__ == '__main__':
    main()
