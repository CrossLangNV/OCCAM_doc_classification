import os

from classifier.data import gen_pdf_paths, pdf2image_preprocessing, gen_im_paths, image_preprocessing
from classifier.models import IMAGE_WIDTH

ROOT = os.path.join(os.path.dirname(__file__), '..')
from PIL import Image


def main(folder_in,
         folder_out,
         shape=(IMAGE_WIDTH, IMAGE_WIDTH),
         fileformat='PDF',
         recursive=True,
         verbose=1):
    """
    For each image in folder, save as image with predescribed shape.
    Returns:

    """

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    def get_images_from_pdf():
        # Go through files and open files
        if verbose:
            n = len([None for _ in gen_pdf_paths(folder_in, recursive=recursive)])

        for i, fp in enumerate(gen_pdf_paths(folder_in, recursive=recursive)):
            if verbose:
                print(f'{i + 1}/{n}')

            l_array_im = list(pdf2image_preprocessing(fp, shape))

            for j, a in enumerate(l_array_im):
                im = Image.fromarray(a)
                yield im

    def get_images_from_images():
        # Go through files and open files
        if verbose:
            n = len([None for _ in gen_im_paths(folder_in, recursive=recursive)])

        for i, fp in enumerate(gen_im_paths(folder_in, recursive=recursive)):
            if verbose:
                print(f'{i + 1}/{n}')

            yield Image.fromarray(image_preprocessing(Image.open(fp), shape))

    if 'pdf' in fileformat.lower():
        gen = get_images_from_pdf()
    elif 'image' in fileformat.lower():
        gen = get_images_from_images()

    for current_index, im in enumerate(gen):
        fp_out = os.path.join(folder_out, f'im_{current_index:04d}.png')
        im.save(fp_out)

    return folder_out


if __name__ == '__main__':
    # Examples:
    if 0:
        # BRIS
        folder_in = os.path.join(ROOT, 'data/raw/BRIS')
        folder_out = os.path.join(ROOT, f'data/preprocessed/BRIS')
    else:
        folder_in = os.path.join(ROOT, 'data/raw/NBB')
        folder_out = os.path.join(ROOT, f'data/preprocessed/NBB')

    # folder_in = os.path.join(ROOT, 'data/test/nbb')
    # folder_out = folder_in

    folder_in = os.path.join(ROOT, 'data/test/official_gazette')
    folder_out = os.path.join(ROOT, 'data/test/official_gazette/prep')
    fileformat = 'image'

    main(folder_in,
         folder_out,
         fileformat=fileformat,
         recursive=False
         )
