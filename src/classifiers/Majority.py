"""
This classifier simply predicts the majority class for each new example.
"""
from .Classifier import Classifier
import operator


class MajorityClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self._classify_function = lambda x: 'CCAT'  # this will get overwritten if trained

    def train(self, data):
        # data will be (document, label pairs, and we just need the most common label)
        labels = data.map(lambda x: (x[1], 1))
        label_counts = labels.reduceByKey(lambda a, b: a + b)
        label_counts = dict(label_counts.collect())

        # extract key with maximum value
        # Extraction function from this SO post: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        majority_class = max(label_counts, key=lambda key: label_counts[key])

        self.has_been_trained = True
        self._classify_function = lambda x: majority_class  # map every instance to the majority class


