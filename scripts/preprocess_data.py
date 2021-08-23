import os
import random

from PIL import Image

from classifier.data import gen_filenames, pdf2image_preprocessing, image_preprocessing
from classifier.models import IMAGE_WIDTH

random.seed(123)

ROOT = os.path.join(os.path.dirname(__file__), '..')


class Main:
    EXT_PDF = '.pdf'
    L_EXT = (EXT_PDF, '.jpg', '.jpeg', '.png', '.tiff', '.tif')

    def __init__(self,
                 folder_in,
                 folder_out,
                 shape=(IMAGE_WIDTH, IMAGE_WIDTH),
                 recursive=True,
                 verbose=1,
                 valid_split: float = .1,
                 test_split: float = .1,
                 ):
        """
        For each image in folder, save as image with the given shape.
        Returns:

        """

        assert valid_split + test_split < 1.0, f'There still has to be room for a training split.' \
                                               f'\nvalid_split={valid_split}\ntest_split={test_split}'

        self.valid_split = valid_split
        self.test_split = test_split

        for idx, (im, subdir) in enumerate(self.image_generator(verbose=verbose, shape=shape)):
            folder_out_save = os.path.join(folder_out, subdir)

            if not os.path.exists(folder_out_save):
                os.makedirs(folder_out_save)

            fp_out = os.path.join(folder_out_save, f'im_{idx:04d}.png')
            im.save(fp_out)

        # def get_images_from_pdf():
        #     # Go through files and open files
        #     if verbose:
        #         n = len([None for _ in gen_pdf_paths(folder_in, recursive=recursive)])
        #
        #     for i, fp in enumerate(gen_pdf_paths(folder_in, recursive=recursive)):
        #         if verbose:
        #             print(f'{i + 1}/{n}')
        #
        #         l_array_im = list(pdf2image_preprocessing(fp, shape))
        #
        #         for j, a in enumerate(l_array_im):
        #             im = Image.fromarray(a)
        #             yield im
        #
        # def get_images_from_images():
        #     # Go through files and open files
        #     if verbose:
        #         n = len([None for _ in gen_im_paths(folder_in, recursive=recursive)])
        #
        #     for i, fp in enumerate(gen_im_paths(folder_in, recursive=recursive)):
        #         if verbose:
        #             print(f'{i + 1}/{n}')
        #
        #         yield Image.fromarray(image_preprocessing(Image.open(fp), shape))
        #
        # def get_images_from_pdf_and_images():
        #     for im in get_images_from_pdf():
        #         yield im
        #     for im in get_images_from_images():
        #         yield im
        #
        # if 'pdf' in fileformat.lower():
        #     gen = get_images_from_pdf()
        # elif 'image' in fileformat.lower():
        #     gen = get_images_from_images()
        # else:
        #     # Do both:
        #     gen = get_images_from_pdf_and_images()
        #
        # for current_index, im in enumerate(gen):
        #
        #     fp_out = os.path.join(folder_out, subdir, f'im_{current_index:04d}.png')
        #     im.save(fp_out)

    def image_generator(self,
                        shape: tuple,
                        verbose: int = 1) -> (str, str):
        """

        Returns:
            yields PIL images with their subdir.
        """

        n = len(_ for _ in gen_filenames(folder_in, recursive=recursive,
                                         ext=self.L_EXT))
        # Go over all the files (PDF's and Images)
        for i_file, filename in enumerate(gen_filenames(folder_in, recursive=recursive,
                                                        ext=self.L_EXT)):
            print(f'File {i_file}/{n} : {filename}')

            subdir = self.get_subdir(valid_split=self.valid_split, test_split=self.test_split)

            # PDF
            if filename.lower().endswith(self.EXT_PDF):
                print('PDF')

                for a in pdf2image_preprocessing(filename, shape, verbose=verbose):
                    im = Image.fromarray(a)
                    yield im, subdir
            else:
                im = Image.fromarray(image_preprocessing(Image.open(filename), shape))
                yield im, subdir

    @staticmethod
    def get_subdir(valid_split,
                   test_split,
                   TRAIN='train',
                   VALID='valid',
                   TEST='test'):
        """
        Decide train - valid - test set

        Args:
           valid_split:
           test_split:

        Returns:
            name of the subdir
        """
        thresh = random.random()

        thresh_valid = 1 - valid_split - test_split
        thresh_test = 1 - test_split

        if thresh >= thresh_test:
            return TEST
        elif thresh >= thresh_valid:
            return VALID
        else:
            return TRAIN


if __name__ == '__main__':
    # Examples:
    recursive = True
    if 0:
        # BRIS
        if 0:
            # BOG
            folder_in = os.path.join(ROOT, 'data/raw/BRIS/BOG')
            folder_out = os.path.join(ROOT, f'data/preprocessed/BOG')

        elif 0:
            # NBB
            folder_in = os.path.join(ROOT, 'data/raw/BRIS/NBB')
            folder_out = os.path.join(ROOT, f'data/preprocessed/NBB')
        elif 0:
            pass
            # folder_in = os.path.join(ROOT, 'data/test/official_gazette')
            # folder_out = os.path.join(ROOT, 'data/test/official_gazette/prep')
            # fileformat = 'image'
            # recursive = False
    elif 1:
        # DH use cases
        # fileformat = ''
        recursive = True
        if 0:
            folder_in = os.path.join(ROOT, 'data/raw/DH/handwritten')
            folder_out = os.path.join(ROOT, 'data/preprocessed/handwritten')

        elif 0:
            folder_in = os.path.join(ROOT, 'data/raw/DH/printed')
            folder_out = os.path.join(ROOT, 'data/preprocessed/printed')

        elif 1:
            folder_in = os.path.join(ROOT, 'data/raw/DH/newspapers')
            folder_out = os.path.join(ROOT, 'data/preprocessed/newspapers')

    Main(folder_in,
         folder_out,
         recursive=recursive,
         )
