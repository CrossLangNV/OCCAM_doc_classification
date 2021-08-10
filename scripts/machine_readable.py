from typing import List, Dict

import numpy as np
from pdfminer.pdfpage import PDFPage
# import chardet
# import PyPDF2

FONT = 'Font'

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
                # TODO
                # _page_full_page_image(page)
                # TODO
                yield False

    l_searchable = list(iter_scanned_page(pdf))

    return np.mean(l_searchable)


def _page_searchable(page: PDFPage):
    return FONT in page.resources.keys()


def _page_full_page_image(page: PDFPage):

    KEY_XOBJECT = 'XObject'
    SUBTYPE = 'Subtype'
    TYPE_IMAGE = 'Image'

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
        return True

    # For a scan, we only expect one x object
    elif len(image_xobjects) != 1:
        return True

    image_xobject = list(image_xobjects.values())[0]

    def get_TODO(page:PDFPage, image: dict):

        # get Mediabox size

        media_box = page.mediabox

        w_media_box = media_box[2] - media_box[0]
        h_media_box = media_box[3] - media_box[1]

        for content_i in page.contents:
            data_content = content_i.resolve().get_data()


        return

    get_TODO(page, image_xobject)

    return




def _get_affine_tf_2(image, content_stream) -> tuple:
    """
    Tries to extract affine transform of the image in contents
    T = [[a, b, e],
         [c, d, f],
         [0, 0, 1]]
    Info about PDF Coordinate Systems on
    https://www.adobe.com/content/dam/acom/en/devnet/acrobat/pdfs/PDF32000_2008.pdf#page=206
    Args:
        image: image element from PyPDF page
        content_stream: content from PyPDF page

    Returns:
        (a, b, c, d, e, f) or None if nothing found
    """
    # Get info of image

    if isinstance(content_stream, bytes):
        l_content_stream = list(map(bytes.strip, content_stream.splitlines()))

        encoding_content = chardet.detect(content_stream)['encoding']

        if encoding_content is None:
            image_bytes = bytes(image, 'utf-8')
        else:
            image_bytes = bytes(image, encoding_content)

    else:
        raise TypeError(type(content_stream))

    # Possibility that content_stream had no linebreaks
    if len(l_content_stream) == 1:
        split = l_content_stream[0].split(image_bytes + b' Do')

        if len(split) == 1:
            return

        elif len(split) == 2:
            assert len(split) <= 2

            *_, a, b, c, d, e, f, cm = split[0].split()

            a, b, c, d, e, f, cm = map(bytes.decode, (a, b, c, d, e, f, cm))

            assert cm == 'cm'

            a, b, c, d, e, f = map(float, (a, b, c, d, e, f))

            co = (a, b, c, d, e, f)
            return co
        else:
            raise ValueError(f"{image_bytes} shouldn't be found more than once")

    # finding image in content stream
    for i, line in enumerate(l_content_stream):
        if line.split()[:2] == [image_bytes, b'Do']:
            co = l_content_stream[i - 1].decode()  # Homogeneous coordinates

            # Not always nicely before and after.
            if 0:
                q = l_content_stream[i - 2].decode()
                Q = l_content_stream[i + 1].decode()

                assert q == 'q'
                assert Q == 'Q'

            a, b, c, d, e, f, cm = co.split()
            assert cm == 'cm'

            a, b, c, d, e, f = map(float, (a, b, c, d, e, f))

            co = (a, b, c, d, e, f)
            return co

    # Found nothing
    return