import os
import random
import warnings
from pathlib import Path
from typing import Union, List, Tuple, Iterator

import numpy as np
from PIL import Image
from pdf2image import convert_from_path

from .models import IMAGE_WIDTH


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

        if not os.path.exists(folder):
            warnings.warn(f"Could not find folder, returning empty list: {folder}", UserWarning)
            # Emtpy array of
            return np.empty(shape=(0,) + shape + (3,))

        if verbose:
            n = len([None for _ in gen_im_paths(folder, recursive=recursive)])

        for i, fp in enumerate(gen_im_paths(folder, recursive=recursive)):
            if verbose:
                print(f'{i + 1}/{n}')

            im = Image.open(fp)
            imgs.append(image_preprocessing(im, shape))

        return np.stack(imgs, axis=0)


def gen_pdf_paths(folder, recursive=True):
    return gen_filenames(folder, recursive, ext='.pdf')


def gen_im_paths(folder, recursive=True):
    return gen_filenames(folder, recursive, ext=('.jpg', '.jpeg', '.png', '.tiff', '.tif'))


def gen_filenames(folder, recursive=True, ext: Union[str, List[str], Tuple[str]] = None) -> Iterator[str]:
    """

    Args:
        folder:
        recursive:
        ext:

    Returns:

    """
    if ext:
        ext = tuple(map(str.lower, ext))

    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = os.path.join(subdir, filename)

            if ext:
                if filepath.lower().endswith(ext):
                    yield filepath
            else:
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
            print(f'Page {i + 1}/{len(pages)}')
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
