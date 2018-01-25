"""
Abstract base class for our document classifiers
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
        Takes an RDD where each entry is a document and returns an RDD of class labels
        :param data: an RDD of unlabeled documents
        :return: an RDD with class labels for the unlabeled documents
        """
        if not self.has_been_trained:
            print("WARNING: Attempting to classify new examples without training the classifier")

        _classify = self._classify_function
        return data.map(lambda doc: _classify(doc))
