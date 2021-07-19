import os
import warnings

import numpy as np
import tensorflow as tf
from PIL import Image

from classifier.methods import sigmoid, FILENAME_MODEL
from classifier.models import IMAGE_WIDTH

ROOT = os.path.join(os.path.dirname(__file__), '..')
TEST_FOLDER = os.path.join(ROOT, r'data/test/NBB')

for filename in [TEST_FOLDER]:
    if not os.path.exists(filename):
        warnings.warn(f"Couldn't find: {filename}", UserWarning)

KEY_LABEL = "label"
KEY_SIGMOID = "sigmoid"


def eval_BRIS(model_BRIS,
              verbose: int = 2) -> dict:
    """
    Evaluates the model on the BRIS test set.

    Args:
        model_BRIS:
        verbose: (int)
            - 0: silent.
            - 1: prints the metrics at the end.
            - 2: Also shows intermediate results
    Returns:
        Dictionary with the metrics
    """

    results = {'label': [],
               'filename': [],
               'sigmoid': []
               }

    def generator_filename_images(folder):
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                yield os.path.join(folder, file)

    n = len([None for _ in generator_filename_images(TEST_FOLDER)])

    for i, filename in enumerate(generator_filename_images(TEST_FOLDER)):

        name = os.path.split(filename)[-1]

        im = Image.open(filename)
        # Default Bicubic
        im_reshape = im.resize((IMAGE_WIDTH, IMAGE_WIDTH))

        y_pred = model_BRIS.predict(np.array(im_reshape)[np.newaxis])
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


def main():
    """

    Returns:

    """

    model_test = tf.keras.models.load_model(FILENAME_MODEL)

    return eval_BRIS(model_test)


if __name__ == '__main__':
    main()
