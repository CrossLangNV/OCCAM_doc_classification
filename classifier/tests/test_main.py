import os
import unittest

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
    def test_upload_image(self):
        headers = {'model-id': "1",
                   }

        with open(FILENAME_IMAGE, 'rb') as f:
            files = {'file': f}



            response = TEST_CLIENT.post("/classify",
                                        headers=headers,
                                        files=files)

        self.assertLess(response.status_code, 300, "Status code should indicate a proper connection.")

        json = response.json()

        for key in ['idx', 'certainty', 'label']:
            with self.subTest('Key %s' % key):
                self.assertIn(key, json, 'Could not retrieve key.')

        return


class TestMultipleFilesClassification(unittest.TestCase):
    def test_single_file_upload(self):

        headers = {'model-id': "1",
                   }

        with open(FILENAME_IMAGE, 'rb') as f:
            files = {'files': f}

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
