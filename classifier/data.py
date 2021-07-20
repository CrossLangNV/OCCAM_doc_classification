import os
import random
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from pdf2image import convert_from_path

from .models import IMAGE_WIDTH

ROOT = os.path.join(os.path.dirname(__file__), '..')
FOLDER_RAW_DATA = os.path.join(ROOT, r'data/raw')
FOLDER_BRIS = os.path.join(FOLDER_RAW_DATA, r'BRIS')
FOLDER_NBB = os.path.join(FOLDER_RAW_DATA, r'NBB')
DATA_PREPROCESSED = os.path.abspath(os.path.join(ROOT, 'data/preprocessed'))

for dir in [DATA_PREPROCESSED,
            FOLDER_NBB,
            FOLDER_BRIS]:
    if not os.path.exists(dir):
        warnings.warn(f'Expected pre-made directory: {dir}', UserWarning)

FILENAME_X = os.path.join(DATA_PREPROCESSED, f'x_training_{IMAGE_WIDTH}.npy')
FILENAME_Y = os.path.join(DATA_PREPROCESSED, f'y_training_{IMAGE_WIDTH}.npy')


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

    def __new__(cls, folder: Path, shape=(IMAGE_WIDTH, IMAGE_WIDTH), verbose=1, *args, **kwargs):
        """
        Walk through folder and add all the images to the stack.

        Args:
            folder:
            shape:
            verbose:
            *args:
            **kwargs:
        """

        imgs = []

        random.seed(123)

        assert os.path.exists(folder), folder

        if verbose:
            n = len([None for _ in gen_im_paths(folder)])

        for i, fp in enumerate(gen_im_paths(folder)):
            if verbose:
                print(f'{i + 1}/{n}')

            im = Image.open(fp)
            imgs.append(image_preprocessing(im, shape))

        return np.stack(imgs, axis=0)


class BRIS(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_BRIS)


class NBB(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_NBB)


def gen_pdf_paths(folder):
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = os.path.join(subdir, filename)

            if filepath.endswith('.pdf'):
                yield filepath


def gen_im_paths(folder):
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = os.path.join(subdir, filename)

            if filepath.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
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
        yield np.array(image_preprocessing(page, shape))


def image_preprocessing(image: Image.Image, shape: tuple) -> np.ndarray:
    """
    Rescale to given shape

    Args:
        image:
        shape:

    Returns:

    """
    if image.size != shape:
        image = image.resize(shape)

    return np.array(image)
