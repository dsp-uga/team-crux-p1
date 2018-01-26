"""
Unit tests for functions in the src.utilities package
"""

import unittest
import pyspark as ps
import src.utilities.preprocess as preprocess
import src.utilities.utils as utils


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        """
        This setup function will be called before any test is run.
        It's a useful place to do initialization that otherwise would have to be repeated for every test function
        """
        pass

    def test_remove_html_character_references(self):
        # test that the function performs as expected
        self.assertEqual(preprocess.remove_html_character_references("&quot;snow"), "snow")
        self.assertEqual(preprocess.remove_html_character_references("desk&quot;"), "desk")
        self.assertEqual(preprocess.remove_html_character_references("airplane&amp;"), "airplane")
        self.assertEqual(preprocess.remove_html_character_references("air&amp;plane"), "airplane")
        self.assertEqual(preprocess.remove_html_character_references("&quot;government&quot;"), "government")
        self.assertEqual(preprocess.remove_html_character_references("Desk&quot;"), "Desk")

        # test that the function has no side effects
        word = "Investment&quot;"
        preprocess.remove_html_character_references(word)
        self.assertEqual(word, "Investment&quot;")

    # TODO: tests for the other preprocessing functions


if __name__ == '__main__':
    unittest.main()
