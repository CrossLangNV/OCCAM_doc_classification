import os
import tempfile
import unittest

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from classifier.main import app

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # ./classifier/

TEST_CLIENT = TestClient(app)

ROOT = os.path.join(os.path.dirname(__file__), '..')
FILENAME_IMAGE = os.path.abspath(os.path.join(ROOT, 'tests/example_files/19154766-page0.jpg'))


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

        for key, value in models.items():
            with self.subTest(f'Model: {key}'):
                self.assertIn('id', value, 'Should contain an ID key')
                self.assertIn('description', value, 'Should contain a description')

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

        for key in ['idx', 'certainty', 'label']:
            with self.subTest('Key %s' % key):
                self.assertIn(key, json, 'Could not retrieve key.')

        return

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

        for key in ['idx', 'certainty', 'label']:
            with self.subTest('Key %s' % key):
                self.assertIn(key, json, 'Could not retrieve key.')

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

        for key in ['idx', 'certainty', 'label']:
            with self.subTest('Key %s' % key):
                self.assertIn(key, json, 'Could not retrieve key.')

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

        for key in ['idx', 'certainty', 'label']:
            with self.subTest('Key %s' % key):
                self.assertIn(key, json0, 'Could not retrieve key.')

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

                for key in ['idx', 'certainty', 'label']:
                    self.assertIn(key, json_i, 'Could not retrieve key.')
