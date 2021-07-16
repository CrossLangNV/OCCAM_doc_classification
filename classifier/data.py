import os
import random
from pathlib import Path

import numpy as np
from pdf2image import convert_from_path

FOLDER_RAW_MEDIA = r'G:\My Drive\OCCAM\media'
FOLDER_BRIS = FOLDER_RAW_MEDIA + r'\BRIS\arne'
FOLDER_MBB = FOLDER_RAW_MEDIA + r'\NBB'

# Commonly used image width for the input of ImageNet
IMAGE_WIDTH = 224

ROOT = os.path.join(os.path.dirname(__file__), '..')
MEDIA = os.path.abspath(os.path.join(ROOT, 'media'))
FILENAME_X = os.path.join(MEDIA, f'x_training_{IMAGE_WIDTH}.npy')
FILENAME_Y = os.path.join(MEDIA, f'y_training_{IMAGE_WIDTH}.npy')


class Training(list):

    def __init__(self, b_scratch=False, *args, **kwargs):
        """

        """

        if b_scratch or (not os.path.exists(FILENAME_X)) or (not os.path.exists(FILENAME_Y)):
            x1 = BRIS()
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


class ImagesFolder(np.ndarray):

    def __new__(cls, folder: Path, shape=(224, 224), *args, **kwargs):
        """
        Walk through folder and add all pdf's as image to stack.

        Args:
            folder:
            shape:
            sampling_prob: Chance to add to data (for subsampling)
            *args:
            **kwargs:
        """

        imgs = []

        random.seed(123)

        # Go through files and open files
        for fp in gen_pdf_paths(folder):
            imgs.extend(pdf2image_preprocessing(fp, shape))

        return np.stack(imgs, axis=0)


class BRIS(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_BRIS)


class NBB(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_MBB)


def gen_pdf_paths(folder):
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = os.path.join(subdir, filename)

            if filepath.endswith('.pdf'):
                yield filepath


def pdf2image_preprocessing(filepath, shape):
    """ Converts a pdf to a list of preprocessed images.
    Preprocessing includes rescaling to a square.

    Args:
        filepath: Filepath to pdf
        shape: Shape of the returned images. tuple of the format (width, height)

    Returns:
        Iterable list of images as numpy arrays.
    """

    pages = convert_from_path(filepath)

    for page in pages:
        page_reshape = page.resize(shape)

        yield np.array(page_reshape)
