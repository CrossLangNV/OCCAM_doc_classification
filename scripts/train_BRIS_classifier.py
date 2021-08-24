import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from classifier.data import ImagesFolder
from classifier.models import DocModel
from scripts.evaluation import eval_BOG, eval_NBB, eval_BOG_VS_NBB

ROOT = os.path.join(os.path.dirname(__file__), '..')
FILENAME_MODEL = os.path.join(ROOT, 'models/model_nbb_bris.h5')


def script_train(dir_label: str,
                 dir_not_label=None,  # Optional
                 label=None,  # TODO optional?
                 output_dir=None,  # TODO optional?
                 output_name='model_classifier_{label}'  # TODO optional?
                 ):
    """

    TODO move to separate file
    Returns:

    """

    assert os.path.exists(dir_label)

    # prepare data
    # generator? with x and y.
    # 1st let's try f (features of x) and noise with label 1 and 0 respectively

    # Init model
    model = DocModel()

    # TODO
    # 541 x 224 x 224 x 3
    x_label = ImagesFolder(dir_label)
    # 541 x 7 x 7 x 1280
    f_label = model.feature(x_label)

    x_not_label = ImagesFolder(dir_not_label)
    f_not_label = model.feature(x_not_label)

    n = f_label.shape[0]
    n_not = f_not_label.shape[0]

    y_label = np.ones((n,))

    y_not_label = np.zeros((n_not,))

    def get_noise_f_y(idx=0):
        # TODO proper distribution
        shape_f_0 = (n,) + (f_label.shape[1:])

        if idx == 0:
            # Just Gaussian noise
            # Model finds this too easy.
            loc = np.mean(f_label)
            scale = np.std(f_label)
        elif idx == 1:
            loc = np.mean(f_label, axis=0, keepdims=True)
            scale = np.std(f_label, axis=0, keepdims=True)
        elif idx == 2:
            # Get per "colour"-channel (average over the spatial dimension)
            loc = np.mean(f_label, axis=(0, 1, 2), keepdims=True)
            scale = np.std(f_label, axis=(0, 1, 2), keepdims=True)

        # we double the scale to have a cloud around the datapoints we have
        f_0 = np.random.normal(loc=loc, scale=2 * scale, size=shape_f_0)
        y_0 = np.zeros((n,))
        return f_0, y_0

    if 1:
        f_comb = np.concatenate([f_label, f_not_label], axis=0)
        y_comb = np.concatenate([y_label, y_not_label], axis=0)
    elif 0:
        f_0, y_0 = get_noise_f_y(2)

        f_comb = np.concatenate([f_label, f_0], axis=0)
        y_comb = np.concatenate([y_label, y_0], axis=0)
    else:
        # Both
        f_0, y_0 = get_noise_f_y(0)

        f_comb = np.concatenate([f_label, f_not_label, f_0], axis=0)
        y_comb = np.concatenate([y_label, y_not_label, y_0], axis=0)

    # Split the data
    f_train, f_valid, y_train, y_valid = train_test_split(f_comb, y_comb, test_size=0.2, shuffle=True)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    model.fit_fast_features(f_train, y_train, epochs=100, verbose=1,
                            validation_data=(f_valid, y_valid),
                            callbacks=es)

    if 0:
        eval_BOG(model, verbose=1)
        eval_NBB(model, verbose=1)
        eval_BOG_VS_NBB(model)

    return model


def main():
    """

    Returns:
    """

    # Get some data
    # TODO Also try to make a user script, and access that one.

    dir_label = os.path.join(ROOT, 'data/preprocessed/BRIS')
    dir_NBB = os.path.join(ROOT, 'data/preprocessed/NBB')
    label = 'BRIS'

    # Train the model
    model = script_train(dir_label=dir_label,
                         dir_not_label=dir_NBB,
                         label=label)

    # Evaluate the model.
    # TODO report somewhere the results/findings
    eval_BOG(model)

    b = 0
    if b:
        if 1:
            model_test = tf.keras.models.load_model(FILENAME_MODEL)
        else:
            model_test = DocModel()
            model_test.load_weights(FILENAME_MODEL)

        eval_BOG(model_test, verbose=1)

    return


if __name__ == '__main__':
    main()
