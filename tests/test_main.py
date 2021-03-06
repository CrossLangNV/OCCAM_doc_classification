import os
import tempfile
import unittest
import warnings

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # ./classifier/

TEST_CLIENT = TestClient(app)

ROOT = os.path.join(os.path.dirname(__file__), '..')
FILENAME_IMAGE = os.path.abspath(os.path.join(ROOT, 'tests/example_files/19154766-page0.jpg'))
# Contains text!
FILENAME_PDF_SCANNED = os.path.abspath(
    os.path.join(ROOT, 'tests/example_files/2006-47600140_scanned.pdf'))
FILENAME_PDF_SCANNED_WITH_TEXT = os.path.abspath(
    os.path.join(ROOT, 'tests/example_files/19136192_scanned_with_mr_text.pdf'))
FILENAME_PDF_MACHINE_READABLE = os.path.abspath(
    os.path.join(ROOT, 'tests/example_files/1999-26101358_machine_readable.pdf'))

if not os.path.exists(FILENAME_IMAGE):
    warnings.warn(f"Couldn't find image: {FILENAME_IMAGE}", UserWarning)


class TestApp(unittest.TestCase):
    def test_root(self):
        """ Test if root url can be accessed
        """

        response = TEST_CLIENT.get("/")

        self.assertLess(response.status_code, 300, "Status code should indicate a proper connection.")


class TestGetModels(unittest.TestCase):
    def test_get(self):
        response = TEST_CLIENT.get("/models")

        self.assertLess(response.status_code, 300, "Status code should indicate a proper connection.")

        models = response.json().get('models')

        for model_info in models:
            with self.subTest(f'Model: {model_info.get("name")}'):

                for key in ['id', 'name', 'description']:
                    self.assertIn(key, model_info.keys(), f'Should contain an {key} key')

    def test_trailing_slash(self):
        response = TEST_CLIENT.get("/models")
        response_slash = TEST_CLIENT.get("/models/")

        self.assertEqual(response.content, response_slash.content, 'Should give same result')


class TestClassification(unittest.TestCase):
    def post_classify(self, file,
                      headers=None
                      ):
        if headers is None:
            headers = {'model-id': "1"}
        files = {'file': file}

        response = TEST_CLIENT.post("/classify",
                                    headers=headers,
                                    files=files)

        with self.subTest('Response'):
            self.assertTrue(response.ok, f"Status code should indicate a proper connection.\n{response}")

        return response.json()

    def test_upload_image(self):

        with open(FILENAME_IMAGE, 'rb') as f:
            json = self.post_classify(f)

        self._check_keys_response(json)

    def test_different_models(self):
        """
        Test the call for different model id's: existing and non-existing.
        Returns:

        """

        response = TEST_CLIENT.get("/models")
        models = response.json().get('models')

        for model in models:
            with self.subTest(model.get('name')):
                model_id = model.get('id')
                headers = {'model-id': str(model_id)}

                with open(FILENAME_IMAGE, 'rb') as f:
                    json = self.post_classify(f, headers=headers)

                self._check_keys_response(json)

        with self.subTest('Non-existing model id'):
            model_id = -1
            headers = {'model-id': str(model_id)}

            with open(FILENAME_IMAGE, 'rb') as f:
                files = {'file': f}

                self.assertRaises(Exception, TEST_CLIENT.post, "/classify", headers=headers,
                                  files=files, msg='Should raise an error')

    def test_grayscale_image(self):
        """
        Single channel image
        :return:
        """

        im_orig = Image.open(FILENAME_IMAGE)

        A_orig = np.array(im_orig)

        # Only cast to uint8 after mean (otherwise over summing is done as uint8's)
        A_gray = np.mean(A_orig, axis=-1).astype(np.uint8)

        im_gray = Image.fromarray(A_gray)

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, 'tmp_gray.png')

            im_gray.save(filename)

            with open(filename, 'rb') as f:
                json = self.post_classify(f)

        self._check_keys_response(json)

    def test_image_grayscale_identical(self):
        """
        4 channel image
        :return:
        """

        im_orig = Image.open(FILENAME_IMAGE)

        A_orig = np.array(im_orig)

        # Only cast to uint8 after mean (otherwise over summing is done as uint8's)
        A_gray = np.mean(A_orig, axis=-1).astype(np.uint8)

        im_gray = Image.fromarray(A_gray)

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, 'tmp_gray.png')

            im_gray.save(filename)

            with open(filename, 'rb') as f:
                json = self.post_classify(f)

        with open(FILENAME_IMAGE, 'rb') as f:
            json_baseline = self.post_classify(f)

        self.assertEqual(json, json_baseline, 'Should give identical results')

    def test_image_alpha(self):
        """
        4 channel image
        :return:
        """

        headers = {'model-id': "1",
                   }

        im_orig = Image.open(FILENAME_IMAGE)

        A_orig = np.array(im_orig)

        # Only cast to uint8 after mean (otherwise over summing is done as uint8's)
        h, w, *c = A_orig.shape

        A_alpha = 255 * np.ones((h, w, 1))
        A_alpha[:, w // 2:, :] = 0  # Make one half transparent

        A_comb = np.concatenate([A_orig, A_alpha], axis=-1).astype(np.uint8)

        im_alpha = Image.fromarray(A_comb)

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, 'tmp_alpha.png')

            im_alpha.save(filename)

            with open(filename, 'rb') as f:
                json = self.post_classify(f)

        self._check_keys_response(json)

    def test_image_alpha_identical(self):
        """
        4 channel image
        :return:
        """

        headers = {'model-id': "1",
                   }

        im_orig = Image.open(FILENAME_IMAGE)

        A_orig = np.array(im_orig)

        # Only cast to uint8 after mean (otherwise over summing is done as uint8's)
        h, w, *c = A_orig.shape

        A_alpha = 255 * np.ones((h, w, 1))
        A_comb = np.concatenate([A_orig, A_alpha], axis=-1).astype(np.uint8)

        im_alpha = Image.fromarray(A_comb)

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, 'tmp_alpha.png')

            im_alpha.save(filename)

            with open(filename, 'rb') as f:
                json = self.post_classify(f)

        with open(FILENAME_IMAGE, 'rb') as f:
            json_baseline = self.post_classify(f)

        self.assertEqual(json, json_baseline, 'Should give identical results')

    def _check_keys_response(self, json):
        for key in ['prediction', 'certainty', 'label']:
            with self.subTest('Key %s' % key):
                self.assertIn(key, json, 'Could not retrieve key.')


class TestMultipleFilesClassification(unittest.TestCase):
    def test_single_file_upload(self):

        headers = {'model-id': "1",
                   }

        with open(FILENAME_IMAGE, 'rb') as f:
            files = [('files', f)]

            response = TEST_CLIENT.post("/classify/multiple",
                                        headers=headers,
                                        files=files)

        self.assertLess(response.status_code, 300, "Status code should indicate a proper connection.")

        json = response.json()

        self.assertEqual(len(json), 1)

        json0 = json[0]

        self._check_keys_response(json0)

    def test_multiple_same_file_upload(self):
        headers = {'model-id': "1",
                   }

        n = 2

        with open(FILENAME_IMAGE, 'rb') as f1, open(FILENAME_IMAGE, 'rb') as f2:
            files = [('files', f1),
                     ('files', f2),
                     ]

            response = TEST_CLIENT.post("/classify/multiple",
                                        headers=headers,
                                        files=files)

        self.assertLess(response.status_code, 300, "Status code should indicate a proper connection.")

        json = response.json()

        self.assertEqual(len(json), n)

        for json_i in json:
            with self.subTest('File %s' % json_i):
                self._check_keys_response(json_i)

    def _check_keys_response(self, json):
        for key in ['prediction', 'certainty', 'label']:
            with self.subTest('Key %s' % key):
                self.assertIn(key, json, 'Could not retrieve key.')


class TestMachineReadable(unittest.TestCase):
    B_DEPRECATED = True

    def setUp(self) -> None:

        if self.B_DEPRECATED:
            warnings.warn('This call is deprecate', DeprecationWarning)

    def test_single_file_upload(self):
        if not self.B_DEPRECATED:
            with open(FILENAME_PDF_SCANNED_WITH_TEXT, 'rb') as f:
                """
                
                """
                files = {'file': f}

                response = TEST_CLIENT.post("/machine_readable",
                                            files=files)

            self.assertEqual(_get_pred_from_response(response), True)

    def test_different_files(self):
        """
        Test for different files

        Returns:

        """
        if not self.B_DEPRECATED:
            def _single_test(filename, b_expected):
                with open(filename, 'rb') as f:
                    files = {'file': f}

                    response = TEST_CLIENT.post("/machine_readable",
                                                files=files)

                self.assertEqual(_get_pred_from_response(response), b_expected, 'Did not expect this prediction')

            for filename, b_expected in {
                FILENAME_PDF_SCANNED: False,
                FILENAME_PDF_SCANNED_WITH_TEXT: True,
                FILENAME_PDF_MACHINE_READABLE: True
            }.items():
                with self.subTest(f'filename: {filename}'):
                    _single_test(filename, b_expected)


class TestScannedDocument(unittest.TestCase):

    def test_different_files(self):
        """
        Test for different files

        Returns:

        """

        def _single_test(filename, b_expected):
            with open(filename, 'rb') as f:
                files = {'file': f}

                response = TEST_CLIENT.post("/scanned_document",
                                            files=files)

            self.assertEqual(_get_pred_from_response(response), b_expected, 'Did not expect this prediction')

        for filename, b_expected in {
            FILENAME_PDF_MACHINE_READABLE: False,
            FILENAME_PDF_SCANNED: True,
            FILENAME_PDF_SCANNED_WITH_TEXT: True,
        }.items():
            with self.subTest(f'filename: {filename}'):
                _single_test(filename, b_expected)


def _get_pred_from_response(response):
    j = response.json()
    p = j.get('prediction')

    return p
