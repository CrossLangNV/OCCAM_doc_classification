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

    # Default Bicubic
    im_reshape = im.resize((IMAGE_WIDTH, IMAGE_WIDTH))
    y_pred = float(model_test.predict(np.array(im_reshape)[np.newaxis]))
    p1 = sigmoid(y_pred)

    return p1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
