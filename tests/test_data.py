import unittest

import numpy as np

from classifier.datasets import Training


class TestData(unittest.TestCase):
    def test_training_data(self):

        x, y = Training(b_scratch=False)

        self.shared_eval(x, y)

    def test_training_data_from_scratch(self):

        x, y = Training(b_scratch=True)

        self.shared_eval(x, y)

    def shared_eval(self, x, y):
        """

        Args:
            x:
            y:

        Returns:

        """
        with self.subTest('Input data'):
            self.assertTrue(len(x), 'X should be non-empty')

            x_0 = x[0]

            self.assertTrue(np.size(x_0), 'X should be non-empty')

            for x_i in x:
                self.assertEqual(np.shape(x_i), np.shape(x_0), 'Every element should have same size.')

        with self.subTest('Output data'):
            self.assertTrue(len(y), 'Y should be non-empty')

            for y_i in y:
                self.assertIsInstance(y_i, (int, np.int, np.int8, np.int16, np.int32, np.int64), 'Every element should be index as integer')

        with self.subTest('Alignment input and output data'):
            self.assertEqual(len(x), len(y), 'Should contain same number of elements')
