import os

import numpy as np
import tensorflow as tf
from PIL import Image

from models import IMAGE_WIDTH

ROOT = os.path.dirname(__file__)


def get_pred_nbb_bris(im: Image) -> float:
    """
    A model that is trained to distinguish BRIS (label 1) from NBB (label 0)
    :param im:
    :return:
    """

    filename_model = os.path.join(ROOT, 'model_nbb_bris.h5')
    model_test = tf.keras.models.load_model(filename_model)

    y_pred = float(model_test.predict(preprocess_image(im)[np.newaxis]))
    p1 = sigmoid(y_pred)

    return p1


def preprocess_image(im: Image):
    """

    :param im:
    :return: a 3D array with UINT8 values
    """
    # To a shape (h, w, 3)

    # Default Bicubic
    im_reshape = im.resize((IMAGE_WIDTH, IMAGE_WIDTH))

    a = np.array(im_reshape)

    # Convert to image with 3 colour channels
    if len(a.shape) == 2:
        # grayscale to 3 channels
        a = np.stack([a] * 3, axis=-1)

    if a.shape[2] == 4:
        a = a[:, :, :3] * (a[:, :, 3:] / 255)

    return a.astype(np.uint8)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
