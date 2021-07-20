import os
import warnings

import numpy as np
import tensorflow as tf
from PIL import Image

from classifier.data import gen_im_paths
from classifier.methods import sigmoid, FILENAME_MODEL
from classifier.models import IMAGE_WIDTH, DocModel

ROOT = os.path.join(os.path.dirname(__file__), '..')
FOLDER_TEST_OFF_GAZ = os.path.join(ROOT, r'data/test/official_gazette')
FOLDER_TEST_NBB = os.path.join(ROOT, r'data/test/nbb')

for filename in [FOLDER_TEST_OFF_GAZ, FOLDER_TEST_NBB]:
    if not os.path.exists(filename):
        warnings.warn(f"Couldn't find: {filename}", UserWarning)

KEY_LABEL = "label"
KEY_SIGMOID = "sigmoid"


def eval_BOG(model_BOG,
             verbose: int = 1) -> dict:
    """
    Evaluates the model on the Belgian Official Gazette test set.

    Args:
        model_BOG:
        verbose: (int)
            - 0: silent.
            - 1: prints the metrics at the end.
            - 2: Also shows intermediate results
    Returns:
        Dictionary with the metrics
    """

    return eval_shared(model_BOG, FOLDER_TEST_OFF_GAZ, verbose=verbose)


def eval_NBB(model_NBB,
             verbose: int = 2) -> dict:
    return eval_shared(model_NBB, FOLDER_TEST_NBB, verbose=verbose)


def eval_shared(model: DocModel,
                dir,
                verbose: int):
    results = {'label': [],
               'filename': [],
               'sigmoid': []
               }

    n = len([None for _ in gen_im_paths(dir)])

    for i, filename in enumerate(gen_im_paths(dir)):

        name = os.path.split(filename)[-1]

        im = Image.open(filename)
        # Default Bicubic
        im_reshape = im.resize((IMAGE_WIDTH, IMAGE_WIDTH))

        y_pred = model.predict(np.array(im_reshape)[np.newaxis])
        y_pred = float(y_pred)  # From array to single value

        if verbose >= 2:
            print(f'{i + 1}/{n} {name}')
            print(f'\ty = {y_pred:.3f}: {"BRIS" if y_pred >= 0 else "Not BRIS"}')
            print('\tSigmoid(y) =', sigmoid(y_pred))

        results.get(KEY_LABEL).append(y_pred >= 0)
        results.get(KEY_SIGMOID).append(sigmoid(y_pred))
        results.get('filename').append(filename)

    acc = np.mean(results.get(KEY_LABEL))
    avg_sigm = np.mean(results.get(KEY_SIGMOID) * 1)

    if verbose:
        print(f'accuracy on test set: {acc:.2%}')
        print(f'avg sigmoid = {avg_sigm:.2%}')

    metrics = {'accuracy': acc,
               'avg sigmoid': avg_sigm}

    return metrics


if __name__ == '__main__':
    model_test = tf.keras.models.load_model(FILENAME_MODEL)

    print('BOG')
    eval_BOG(model_test, verbose=1)
    print('NBB')
    eval_NBB(model_test, verbose=1)
