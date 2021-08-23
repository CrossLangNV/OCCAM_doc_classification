import os
import warnings

import numpy as np

from classifier.data import ImagesFolder
from classifier.models import IMAGE_WIDTH

ROOT = os.path.join(os.path.dirname(__file__), '..')
FOLDER_PREPROCESSED_DATA = os.path.join(ROOT, r'data/preprocessed')
# BRIS
FOLDER_BOG = os.path.join(FOLDER_PREPROCESSED_DATA, r'BOG')
FOLDER_NBB = os.path.join(FOLDER_PREPROCESSED_DATA, r'NBB')
# DH
FOLDER_NEWSPAPERS = os.path.join(FOLDER_PREPROCESSED_DATA, r'newspapers')
FOLDER_PRINTED = os.path.join(FOLDER_PREPROCESSED_DATA, r'printed')
FOLDER_HANDWRITTEN = os.path.join(FOLDER_PREPROCESSED_DATA, r'handwritten')
SUBDIR_TRAIN = 'train'
SUBDIR_VALID = 'valid'
SUBDIR_TEST = 'test'

FOLDER_FEATURES = os.path.join(ROOT, r'data/features')
FILENAME_X = os.path.join(FOLDER_FEATURES, f'x_training_{IMAGE_WIDTH}.npy')
FILENAME_Y = os.path.join(FOLDER_FEATURES, f'y_training_{IMAGE_WIDTH}.npy')

for dir in [FOLDER_FEATURES,
            FOLDER_NBB,
            FOLDER_BOG]:
    if not os.path.exists(dir):
        warnings.warn(f'Expected pre-made directory: {dir}', UserWarning)


class Training(list):

    def __init__(self, b_scratch=False, *args, **kwargs):
        """

        """

        if b_scratch or (not os.path.exists(FILENAME_X)) or (not os.path.exists(FILENAME_Y)):
            x1 = BOG()
            y1 = [1 for _ in x1]

            x2 = NBB()
            y2 = [0 for _ in x2]

            x = np.concatenate([x1, x2], axis=0)
            y = np.concatenate([y1, y2], axis=0)

            np.save(FILENAME_X, x)
            np.save(FILENAME_Y, y)

        x = np.load(FILENAME_X)
        y = np.load(FILENAME_Y)

        super(Training, self).__init__([x, y])


class BOG(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_BOG)


class NBB(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_NBB)


class Newspapers(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_NEWSPAPERS)


class Printed(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_PRINTED)


class Handwritten(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_HANDWRITTEN)

# Bigger collections
class BRIS(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        if_BOG = BOG()
        if_NBB = NBB()

        c = np.concatenate([if_BOG, if_NBB], axis=0)
        return c


class DH(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        if_news = Newspapers()
        if_print = Printed()
        if_handwritten = Handwritten()

        c = np.concatenate([if_news, if_print, if_handwritten], axis=0)
        return c