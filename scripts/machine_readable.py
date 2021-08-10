from typing import List, Dict

import numpy as np
from pdfminer.pdfpage import PDFPage

from machine_readable.pdf_processing import get_affine_tf_co_page

DELTA = 0.01  # Allowed dissimilarity

FONT = 'Font'
KEY_XOBJECT = 'XObject'
SUBTYPE = 'Subtype'
TYPE_IMAGE = 'Image'


def machine_readable_classifier(pdf: str) -> bool:
    """
    Classifies the machine readability of a pdf.
    Machine readability is defined as if every page of the PDF is searachable.

    Args:
        pdf:
            Filepath to a pdf

    Returns:
        bool: True if machine readable, False if not.
    """

    l = list(pdf_searchable_pages(pdf))

    return all(l)


def pdf_searchable_pages(pdf) -> List[bool]:
    """

    Args:
        pdf: Path to pdf

    Returns:
        List of
    """

    with open(pdf, 'rb') as infile:
        for page in PDFPage.get_pages(infile):
            b_searchable = FONT in page.resources.keys()
            yield b_searchable


def _mr_classifier_file(file):
    """

    Args:
        file: e.g. >> with open(pdf, 'rb') as infile: __mr_classifier_file(file)

    Returns:

    """

    l = list(_pdf_searchable_pages(file))

    return all(l)


def _p_machine_readable(pdf) -> float:
    l = list(_pdf_searchable_pages(pdf))

    return np.mean(l)


def _pdf_searchable_pages(file):
    """
    Detect for each page if it contains text.

    Args:
        file:

    Returns:
        an iterator
    """
    for page in PDFPage.get_pages(file):
        b_searchable = FONT in page.resources.keys()
        yield b_searchable


def scanned_document(pdf):
    """ Predict the likelihood that a PDF file might contain scanned documents.

    Args:
        pdf: opened PDF file

    Returns:

    Examples:
        Example of retrieving the likelihood that the file 'path_to_pdf' is a scanned document.

        >>> with open(path_to_pdf) as f: print(scanned_document(f))
        .95

    """

    def iter_scanned_page(pdf):

        for page in PDFPage.get_pages(pdf):
            searchable = _page_searchable(page)

            # No text, thus we assume it's scanned (could be an empty page, but we'll ignore that for now)
            if not searchable:
                yield True

            else:
                # Even when there is text, it could be that there is an scan in the background
                yield _page_full_page_image(page)

    l_scanned = list(iter_scanned_page(pdf))

    return np.mean(l_scanned)


def _page_searchable(page: PDFPage):
    return FONT in page.resources.keys()


def _page_full_page_image(page: PDFPage) -> bool:
    def get_image_xobjects(page: PDFPage) -> Dict[str, Dict]:
        """
        Find all image xobjects in the pdf page
        Args:
            page: PDFPage

        Returns:
            image xObjects
        """

        xobjects = page.resources.get(KEY_XOBJECT, {})

        image_objects = {}
        for name, ref_i in xobjects.items():

            stream_i = ref_i.resolve()
            image_attrs = stream_i.attrs
            if TYPE_IMAGE != image_attrs.get(SUBTYPE).name:
                continue

            image_objects[name] = image_attrs

        return image_objects

    image_xobjects = get_image_xobjects(page)

    # no images
    if len(image_xobjects) == 0:
        return False

    # For a scan, we only expect one x object
    elif len(image_xobjects) != 1:
        return False

    image_xobject = list(image_xobjects.values())[0]

    def check_similar_page(page: PDFPage, image: dict, image_name,
                           thresh=DELTA) -> bool:
        """

          Args:
              w_page:
              h_page:

          Returns:
              True if similar, False if dissimilar
          """

        # get Mediabox size

        media_box = page.mediabox

        w_media_box = media_box[2] - media_box[0]
        h_media_box = media_box[3] - media_box[1]

        co_image = get_affine_tf_co_page(image_name, page)

        def get_T(a=1, b=0, c=0, d=1, e=0, f=0):
            T = np.array([[a, b, e],
                          [c, d, f],
                          [0, 0, 1]])

            return T

        def _affine_transform(co, T):
            """
            Returns (x, y) transformed
            Args:
                co:
                T:

            Returns:

            """

            assert np.shape(co) == (2,)

            # (x, y, 1)
            co3 = np.concatenate([co, [1]])

            return np.dot(T, co3)[:2]

        T = get_T(*co_image)
        T_reverse = get_T(a=1 / w_media_box, d=1 / h_media_box)

        left_top = (0, 0)
        right_top = (1, 0)
        left_bot = (0, 1)
        right_bot = (1, 1)

        for co_i in [left_top, right_top, left_bot, right_bot]:
            co_i_star = _affine_transform(_affine_transform(co_i, T), T_reverse)

            # Check if the coordinates are similar after transformation
            if np.linalg.norm(co_i_star - co_i) >= thresh:
                # This image does not cover the whole page
                return False

        # Everything checked out
        return True

    return check_similar_page(page, image_xobject, image_name=list(image_xobjects)[0])
