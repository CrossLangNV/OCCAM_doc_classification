import os
import unittest

from scripts.machine_readable import machine_readable_classifier

ROOT = os.path.join(os.path.dirname(__file__), '../..')


class TestMRClassifier(unittest.TestCase):

    def test_classifier(self):
        """
        """

        l_pdf_machine_readable = ['data/test/nbb/2001-25304663.pdf']

        l_pdf_non_machine_readable = ['data/test/nbb/1999-07403130.pdf']

        def sub_test(fp, b_mr: bool):

            b_pred = machine_readable_classifier(fp)

            self.assertEqual(b_pred, b_mr, "Unexpected classification.")

        for mr in l_pdf_machine_readable:
            with self.subTest(f'MR file: {mr}'):
                fp = os.path.join(ROOT, mr)
                sub_test(fp, True)

        for non_mr in l_pdf_non_machine_readable:
            with self.subTest(f'non MR file: {non_mr}'):
                fp = os.path.join(ROOT, non_mr)
                sub_test(fp, False)
