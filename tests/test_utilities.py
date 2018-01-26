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

    # TODO: tests for the other preprocessing functions


class TestUtils(unittest.TestCase):
    def setUp(self):
        # for some reason, running spark code within a unittest throws a bunch of ResourceWarnings
        # check out this issue: https://github.com/requests/requests/issues/3912
        warnings.filterwarnings(action="ignore", category=ResourceWarning)
        pass

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
