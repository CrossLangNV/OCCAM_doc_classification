import os
import warnings

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix
from classifier.data import gen_im_paths, image_preprocessing, ImagesFolder
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

    return eval_shared(model_BOG, FOLDER_TEST_OFF_GAZ, label='BOG', verbose=verbose)


def eval_NBB(model_NBB,
             verbose: int = 1) -> dict:
    return eval_shared(model_NBB, FOLDER_TEST_NBB, label='NBB', index = 0, erbose=verbose)

def eval_BOG_VS_NBB(model,
                    verbose: int = 1):
    """

    Args:
        model_BOG:
        verbose:

    Returns:
        a confusion matrix.
    """

    x_BOG = ImagesFolder(FOLDER_TEST_OFF_GAZ, recursive=False)
    x_NBB = ImagesFolder(FOLDER_TEST_NBB)

    y_BOG = np.ones((x_BOG.shape[0]))
    y_NBB = np.zeros((x_NBB.shape[0]))

    x = np.concatenate([x_BOG,x_NBB ] ,axis=0)
    y= np.concatenate([y_BOG,y_NBB ],axis=0)

    y_pred = model.predict(x)

    y_pred_label = (y_pred >= 0).astype(int)


    conf = confusion_matrix(y, y_pred_label, )

    if verbose:
        print(conf)

    return conf


def eval_shared(model: DocModel,
                dir,
                label: str,
                index = 1,
                verbose: int = 1):
    results = {'label': [],
               'filename': [],
               'sigmoid': []
               }

    n = len([None for _ in gen_im_paths(dir)])

    for i, filename in enumerate(gen_im_paths(dir)):

        name = os.path.split(filename)[-1]

        im = image_preprocessing(Image.open(filename))

        y_pred = model.predict(np.array(im)[np.newaxis])
        y_pred = float(y_pred)  # From array to single value

        b_corr = y_pred >= 0
        s_pred = sigmoid(y_pred)
        if index == 0:
            b_corr = not b_corr
            s_pred = 1 - s_pred

        if verbose >= 2:

            print(f'{i + 1}/{n} {name}')
            print(f'\ty = {y_pred:.3f}: {f"{label}" if b_corr else f"Not {label}"}')
            print(f'\tsigmoid(y) = {s_pred:.2%}')

        results.get(KEY_LABEL).append(b_corr)
        results.get(KEY_SIGMOID).append(s_pred)
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

    # print('BOG')
    # eval_BOG(model_test, verbose=1)
    # print('NBB')
    # eval_NBB(model_test, verbose=1)
    print("BOG vs. NB")
    eval_BOG_VS_NBB(model_test, verbose=1)
