"""
This file defines an interface for our document classifiers
"""


class Classifier:
    def __init__(self):
        self._classify_function = None  # function used to classify a single example - will be defined after training
        self.has_been_trained = False

    def train(self, data):
        """
        The train method allows the classifier to learn from a set of labeled data
        The data passed should be an RDD containing (example, class) pairs

        :param data: a Spark RDD containing (example, class) pairs
        :return: None
        """
        pass

    def classify(self, data):
        """
        The classify method accepts a unlabeled dataset and returns a new dataset containing the predicted classes
        The train method should always be called before classify

        :param data: a Spark RDD where each entry is an unlabeled example
        :return: a Spark RDD containing the predicted class of each item in the original dataset
        """
        pass
