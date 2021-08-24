import json
from typing import Tuple, Optional
from pdfminer.pdfpage import PDFPage

def get_affine_tf_co_page(image_name: str, page: PDFPage):
    """
    Goes through the page content to find

    Args:
        image_name:
        page:

    Returns:

    """

    for content_i in page.contents:
        data_content = content_i.resolve().get_data()
        co = get_affine_tf_co(image_name, data_content)

        if co:
            return co

    raise ValueError(f"Couldn't find {image_name} in the PDF page")

def get_affine_tf_co(image_name: str, content_stream: bytes) -> Optional[Tuple[float]]:
    """
    Tries to extract affine transform of the image in contents
    T = [[a, b, e],
         [c, d, f],
         [0, 0, 1]]
    Info about PDF Coordinate Systems on
    https://www.adobe.com/content/dam/acom/en/devnet/acrobat/pdfs/PDF32000_2008.pdf#page=206
    Args:
        image_name: image element name from a page. e.g. "Bg" or "im1"
        content_stream: content from PyPDF page

    Returns:
        (a, b, c, d, e, f) or None if nothing found
    """
    # Get info of image

    if not isinstance(content_stream, bytes):
        raise TypeError(f"Expected bytes for content_stream: {type(content_stream)}")

    # splitlines + strip
    lines_content_stream = list(map(bytes.strip, content_stream.splitlines()))

    encoding_content = json.detect_encoding(content_stream)
    if not encoding_content:
        encoding_content = 'utf-8'

    # Remove leading / if there
    bytes_image_name = image_name.strip().lstrip('/').encode(encoding_content)

    # look for b"{image_bytes} Do"
    # Has something in the form of
    # b"/im1 Do"
    search = b'/' + bytes_image_name + b' Do'

    def get_co_from_line(co_line):
        *_, a, b, c, d, e, f, cm = co_line.split()

        a, b, c, d, e, f, cm = map(lambda ch: ch.decode(encoding_content), (a, b, c, d, e, f, cm))
        assert cm == 'cm'

        co = tuple(map(float, (a, b, c, d, e, f)))
        return co

    # Possibility that content_stream had no linebreaks
    if len(lines_content_stream) == 1:
        line_content_stream = lines_content_stream[0]
        split_search = line_content_stream.split(search)

        if len(split_search) == 2:
            co_line = split_search[0]
            return get_co_from_line(co_line)

        elif len(split_search) == 1:
            # Did not find 'search'
            return

        else:
            raise ValueError(f"{bytes_image_name} shouldn't be found more than once in {line_content_stream}")

    for prev_line, line in zip(lines_content_stream[:-1], lines_content_stream[1:]):

        if line.split()[:2] == [b'/' + bytes_image_name, b'Do']:

            return get_co_from_line(prev_line)

        else:
            continue
