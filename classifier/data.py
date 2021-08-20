import os
import random
import warnings
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from pdf2image import convert_from_path

from .models import IMAGE_WIDTH

ROOT = os.path.join(os.path.dirname(__file__), '..')
FOLDER_PREPROCESSED_DATA = os.path.join(ROOT, r'data/preprocessed')
FOLDER_FEATURES = os.path.join(ROOT, r'data/features')
FOLDER_BOG = os.path.join(FOLDER_PREPROCESSED_DATA, r'BOG')
FOLDER_NBB = os.path.join(FOLDER_PREPROCESSED_DATA, r'NBB')

for dir in [FOLDER_FEATURES,
            FOLDER_NBB,
            FOLDER_BOG]:
    if not os.path.exists(dir):
        warnings.warn(f'Expected pre-made directory: {dir}', UserWarning)

FILENAME_X = os.path.join(FOLDER_FEATURES, f'x_training_{IMAGE_WIDTH}.npy')
FILENAME_Y = os.path.join(FOLDER_FEATURES, f'y_training_{IMAGE_WIDTH}.npy')


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

    def __new__(cls, folder: Union[str, Path], shape=(IMAGE_WIDTH, IMAGE_WIDTH), verbose=1, recursive=True, *args,
                **kwargs):
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
            n = len([None for _ in gen_im_paths(folder, recursive=recursive)])

        for i, fp in enumerate(gen_im_paths(folder, recursive=recursive)):
            if verbose:
                print(f'{i + 1}/{n}')

            im = Image.open(fp)
            imgs.append(image_preprocessing(im, shape))

        return np.stack(imgs, axis=0)


class BRIS(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_BOG)


class NBB(ImagesFolder):
    def __new__(cls, *args, **kwargs):
        return ImagesFolder.__new__(cls, folder=FOLDER_NBB)


def gen_pdf_paths(folder, recursive=True):
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = os.path.join(subdir, filename)

            if filepath.endswith('.pdf'):
                yield filepath
        if not recursive:  # Finished after first next()
            break


def gen_im_paths(folder, recursive=True):
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = os.path.join(subdir, filename)

            if filepath.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                yield filepath
        if not recursive:  # Finished after first next()
            break


def pdf2image_preprocessing(filepath, shape, verbose=1):
    """ Converts a pdf to a list of preprocessed images.
    Preprocessing includes rescaling to a square.

    Args:
        filepath: Filepath to pdf
        shape: Shape of the returned images. tuple of the format (width, height)

    Returns:
        Iterable list of images as numpy arrays.
    """

    # immediately change size to limit memory.
    pages = convert_from_path(filepath,
                              size=shape)

    for i, page in enumerate(pages):
        if verbose:
            print(f'Page {i+1}/{len(pages)}')
        yield image_preprocessing(page, shape)


def image_preprocessing(image: Image.Image, shape: tuple) -> np.ndarray:
    """
    Rescale to given shape

    Args:
        image:
        shape:

    Returns:

    """
    if image.size != shape:
        # Default Bicubic
        image = image.resize(shape)

    return np.array(image)
