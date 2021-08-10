import unittest

from machine_readable.pdf_processing import get_affine_tf_co


class TestAffineTF(unittest.TestCase):

    def test_no_content_stream(self):
        """
        If image name not contained in contained in content stream it should return None
        """

        image = '/Bg'

        with self.subTest('Short string'):
            content_stream = b'q'

            tf_co = get_affine_tf_co(image, content_stream)

            self.assertIsNone(tf_co)

        with self.subTest('Long string'):
            content_stream = b'q/Span<</MCID 0>>BDC EMC/Artifact<</Type/Pagination/Subtype/Header>>BDC/F_2 9.500 Tf BT 387.60 824.88 TD[(\x1e)300(\x01)-534(")243(\x01)-55(\x06)178(\x01)-255(#)195(\x01)-382( )]TJ'

            tf_co = get_affine_tf_co(image, content_stream)

            self.assertIsNone(tf_co)

    def test_content_stream(self):
        """
        Content stream with line breaks
        """

        image = '/Bg'

        with self.subTest('\\n'):
            content_stream = b'q\n/Artifact<</Type/Page>>BDC\n591.35 0 0 840.50 0 -0.02000 cm\n/Bg Do\n EMC\n Q '
            tf_co_n = get_affine_tf_co(image, content_stream)
            self.assertIsNotNone(tf_co_n, 'coordinates should be found')

        with self.subTest('\\r'):
            content_stream = b'q\r/Artifact<</Type/Page>>BDC\r591.35 0 0 840.50 0 -0.02000 cm\r/Bg Do\r EMC\r Q '
            tf_co_r = get_affine_tf_co(image, content_stream)
            self.assertIsNotNone(tf_co_n, 'coordinates should be found')

        with self.subTest('\\r\\n'):
            content_stream = b'q\r\n/Artifact<</Type/Page>>BDC\r\n591.35 0 0 840.50 0 -0.02000 cm\r\n/Bg Do\r\n EMC\r\n Q '
            tf_co_rn = get_affine_tf_co(image, content_stream)
            self.assertIsNotNone(tf_co_n, 'coordinates should be found')

        self.assertEqual(tf_co_n, tf_co_r, 'outputs are expected to be equal, independent of linebreaks')
        self.assertEqual(tf_co_n, tf_co_rn, 'outputs are expected to be equal, independent of linebreaks')

    def test_content_stream_no_space(self):
        """
        Content stream doesn't always contain line breaks
        """

        content_stream = b'q/Artifact<</Type/Page>>BDC 591.35 0 0 840.50 0 -0.02000 cm/Bg Do EMC Q '
        image = '/Bg'

        tf_co = get_affine_tf_co(image, content_stream)

        self.assertIsNotNone(tf_co, 'coordinates should be found')

    def test_example_10_08_2021(self):

        content_stream = b'q\r591.480 0 0 842.040 0 -0.120 cm \r/im1 Do\rQ \r'
        image_name = 'im1'

        tf_co = get_affine_tf_co(image_name, content_stream)
        self.assertIsNotNone(tf_co, 'coordinates should be found')