import unittest

import numpy as np

from classifier.models import DocModel, IMAGE_WIDTH


class TestDocNet(unittest.TestCase):

    def test_init(self):
        """ Test if model can be initialised and compiles without problems.
        """

        net = DocModel()

        self.assertTrue(net)

    def test_features(self):
        net = DocModel()

        x = np.random.rand(IMAGE_WIDTH, IMAGE_WIDTH, 3)

        f = net.feature(x)

        self.assertEqual(f.shape, (1, 7, 7, 1280), 'Expected feature shape is of size (1, 7, 7, 1280).')

    def test_prediction(self):
        """ Test

        Returns:

        """

        BATCH_SIZE = 10
        net = DocModel()

        x = np.random.randint(0, 255, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, 3))

        y = net.predict(x)

        self.assertEqual(y.shape, (BATCH_SIZE, 1), 'Expected output shape is of size (n, 1), with n the batch size.')

    def test_train(self):
        """ Test that the default training still works and that it's precompiled.

        Returns:

        """

        BATCH_SIZE = 10
        net = DocModel()

        x, y = (np.random.randint(0, 256, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, 3)),
                np.random.randint(0, 2, (BATCH_SIZE, 1)))

        hist = net.fit(x, y, epochs=2)

        y_pred = net.predict(x)

    def test_train_fast(self):
        """ Test if improved/faster fitting works.
        Check validation data. Loss on training is semi-random due to dropout.
        """
        BATCH_SIZE = 10
        net = DocModel()

        x, y = (np.random.randint(0, 256, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, 3)),
                np.random.randint(0, 2, (BATCH_SIZE, 1)))

        hist = net.fit_fast(x, y, validation_data=(x, y), epochs=2, batch_size=BATCH_SIZE,
                            )

        y_pred = net.predict(x)

        hist_ = hist.history.get('val_loss')
        self.assertTrue(len(hist_) > 1, 'Sanity check that model has trained for more than one step.')
        self.assertLess(hist_[-1], hist_[0], 'loss should have improved')


if __name__ == '__main__':
    unittest.main()
