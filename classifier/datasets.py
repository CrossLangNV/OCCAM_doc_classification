import os
import warnings
from enum import Enum, unique
from typing import List

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

FOLDER_FEATURES = os.path.join(ROOT, r'data/features')
FILENAME_X = os.path.join(FOLDER_FEATURES, f'x_training_{IMAGE_WIDTH}.npy')
FILENAME_Y = os.path.join(FOLDER_FEATURES, f'y_training_{IMAGE_WIDTH}.npy')

for dir in [FOLDER_FEATURES,
            FOLDER_NBB,
            FOLDER_BOG]:
    if not os.path.exists(dir):
        warnings.warn(f'Expected pre-made directory: {dir}', UserWarning)


@unique
class Subset(Enum):
    """
    The different subsets of data: training, validating and testing
    """
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


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
    def __new__(cls, subset: Subset, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=os.path.join(FOLDER_BOG, subset.value))


class NBB(ImagesFolder):
    def __new__(cls, subset: Subset, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=os.path.join(FOLDER_NBB, subset.value))


class Newspapers(ImagesFolder):
    def __new__(cls, subset: Subset, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=os.path.join(FOLDER_NEWSPAPERS, subset.value))


class Printed(ImagesFolder):
    def __new__(cls, subset: Subset, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=os.path.join(FOLDER_PRINTED, subset.value))


class Handwritten(ImagesFolder):
    def __new__(cls, subset: Subset, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=os.path.join(FOLDER_HANDWRITTEN, subset.value))


class ImageFolderCollection(ImagesFolder):

    @staticmethod
    def filter_empty_if(l_if: List[ImagesFolder]) -> List[ImagesFolder]:
        # Filter emtpy arrays
        return [if_ for if_ in l_if if if_.size]


# Bigger collections
class BRIS(ImageFolderCollection):
    def __new__(cls, subset: Subset, *args, **kwargs):
        if_BOG = BOG(subset)
        if_NBB = NBB(subset)

        c = np.concatenate(cls.filter_empty_if([if_BOG, if_NBB]), axis=0)
        return c


class DH(ImageFolderCollection):
    def __new__(cls, subset: Subset, *args, **kwargs):
        if_news = Newspapers(subset)
        if_print = Printed(subset)
        if_handwritten = Handwritten(subset)

        c = np.concatenate(cls.filter_empty_if([if_news, if_print, if_handwritten]), axis=0)
        return c
