"""
Unit tests for functions in the src.utilities package
"""

import unittest
import warnings
from pyspark import SparkContext, SparkConf
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

    def test_strip_punctuation(self):
        # test that the function performs as expected
        self.assertEqual(preprocess.strip_punctuation("'snow"), "snow")
        self.assertEqual(preprocess.strip_punctuation("snow."), "snow")
        self.assertEqual(preprocess.strip_punctuation("snow!"), "snow")
        self.assertEqual(preprocess.strip_punctuation("?snow?"), "snow")
        self.assertEqual(preprocess.strip_punctuation("snow\""), "snow")
        self.assertEqual(preprocess.strip_punctuation("sn!ow"), "snow")

        # test that the function has no side effects
        word = "Investment."
        preprocess.remove_html_character_references(word)
        self.assertEqual(word, "Investment.")

    def test_tokenize(self):
        # test that the function performs as expected
        line = "the quick   brown   \t  fox  jumps \r\n over the lazy \n dog"
        tokens = preprocess.tokenize(line)

        self.assertTrue("the" in tokens)
        self.assertTrue("quick" in tokens)
        self.assertTrue("brown" in tokens)
        self.assertTrue("fox" in tokens)
        self.assertTrue("jumps" in tokens)
        self.assertTrue("over" in tokens)
        self.assertTrue("the" in tokens)
        self.assertTrue("lazy" in tokens)
        self.assertTrue("dog" in tokens)

    def test_split_by_comma(self):
        line = "the,quick,brown,fox,jumps,over,the,lazy,dog"
        tokens = preprocess.split_by_comma(line)

        self.assertTrue("the" in tokens)
        self.assertTrue("quick" in tokens)
        self.assertTrue("brown" in tokens)
        self.assertTrue("fox" in tokens)
        self.assertTrue("jumps" in tokens)
        self.assertTrue("over" in tokens)
        self.assertTrue("the" in tokens)
        self.assertTrue("lazy" in tokens)
        self.assertTrue("dog" in tokens)

    def test_remove_irrelevant_labels(self):
        labels = ["GCAT", "CCAT", "ECAT", "MCAT", "E12", "E54" "G154", "M13", "GWEA"]
        filtered = preprocess.remove_irrelevant_labels(labels)

        self.assertEqual(len(filtered), 4)
        self.assertTrue("GCAT" in filtered)
        self.assertTrue("CCAT" in filtered)
        self.assertTrue("ECAT" in filtered)
        self.assertTrue("MCAT" in filtered)


class TestUtils(unittest.TestCase):
    def setUp(self):
        # for some reason, running spark code within a unittest throws a bunch of ResourceWarnings
        # check out this issue: https://github.com/requests/requests/issues/3912
        warnings.filterwarnings(action="ignore", category=ResourceWarning)

    def test_custom_zip(self):
        # for this test, we don't want to run on an actual cluster.  A local master is sufficient
        conf = SparkConf().setAppName("Unit Tests").setMaster("local").set('spark.logConf', 'true')
        sc = SparkContext(conf=conf)
        sc.setLogLevel("FATAL")

        nums = list(range(0, 10))
        squares = [num**2 for num in nums]
        pairs = [(num, num**2) for num in nums]

        # the custom zip function should work on RDDs with different numbers of slices
        rdd1 = sc.parallelize(nums, 5)
        rdd2 = sc.parallelize(squares, 3)
        combined = utils.custom_zip(rdd1, rdd2)
        combined = combined.sortByKey()
        combined = list(combined.collect())

        for idx, tuple in enumerate(pairs):
            self.assertEqual(tuple, combined[idx])


if __name__ == '__main__':
    unittest.main()
