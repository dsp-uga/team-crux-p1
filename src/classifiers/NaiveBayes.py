import src.preprocess as preprocess
from .Classifier import Classifier
import pyspark
import numpy as np


class NaiveBayesClassifier(Classifier):
    def __init__(self, spark_context, stopwords=[]):
        """
        This classifier using multinominal Naive Bayes to classify documents from the Reuters Corpus
        This classifier assumes that the program is running in an Apache Spark cluster
        :param spark_context: the Spark context object in which the RDD operations are being performed
        :param stopwords: file containing stopwords to be removed from the corpus
        """
        Classifier.__init__(self)
        self.sc = spark_context
        self._classify_function = lambda x: 'CCAT'  # default classification function

        # mapping from class label to numeric index
        self.CLASSES = self.sc.broadcast({
            "CCAT": 0,
            "ECAT": 1,
            "GCAT": 2,
            "MCAT": 3
        })
        # mapping from numeric index back to class label
        self.CLASS_INDICES = self.sc.broadcast({v:k for k,v in self.CLASSES.value.items()})

        # read list of stopwords:
        self.SW = self.sc.broadcast(stopwords)

    def train(self, data):
        """
        Trains the classifier using the specified set of labeled documents
        Builds a private function to classify a single document

        :param data: RDD containing ( document_contents, document_label ) pairs
        :return: None
        """
        assert(isinstance(data, pyspark.RDD))
        TOTAL_DOCS = self.sc.broadcast(data.count())
        # get the number of documents per class
        class_unit_counts = data.map(lambda x: (x[1], 1))  # (class_label, 1) tuples
        class_counts = class_unit_counts.reduceByKey(lambda a, b: a + b)

        # we can now calculate class probabilities
        class_probabilities = class_counts.mapValues(lambda x: x/TOTAL_DOCS.value)
        CLASS_PROBABILITY = self.sc.broadcast(dict(class_probabilities.collect()))

        # tokenize and preprocess documents:
        _SW = self.SW
        processed = data.map(lambda tuple: (tuple[1], tuple[0]) )  # temporarily swap to (class, document_contents)
        words = processed.flatMapValues(preprocess.tokenize)  # (class, word) pairs
        words = words.mapValues(preprocess.remove_html_character_references)
        words = words.mapValues(preprocess.strip_punctuation)
        words = words.mapValues(lambda x: x.lower())
        words = words.filter(lambda x: len(x[1]) > 0)  # make sure
        words = words.filter(lambda x: x[1] not in _SW.value)  # remove stopwords from corpus
        words.cache()

        # extract the vocabulary size (number of unique words in the corpus)
        vocab = words.map(lambda x: x[1]).distinct()
        VOCAB_SIZE = self.sc.broadcast(vocab.count())

        # we can now count the number of words (non-unique) in each class
        class_words = words.map(lambda x: (x[0], 1))  # (class, 1) tuples for each word in the corpus
        class_words = class_words.reduceByKey(lambda a, b: a + b)
        # CLASS_WORD_COUNTS maps class c to the total number of words in all docs of class c
        CLASS_WORD_COUNTS = self.sc.broadcast(dict(class_words.collect()))

        # now we need to count up how many times each word appears in each class
        _CLASSES = self.CLASSES
        _CLASS_INDICES = self.CLASS_INDICES

        def _word_tuple_to_class_vec(x):
            """
            Takes a (class, word) tuple and converts it to a (word, class_vector) tuple
            Class vector is a length-n vector where n is the number of classes
            All values of the class vector will be zero except for the position corresponding the the class
            For example, ('CCAT', 'foobar') will be converted to ('foobar', [1, 0, 0, 0])
            """
            class_label, word = x
            class_index = _CLASSES.value[class_label]
            class_vector = np.zeros(len(_CLASSES.value))
            class_vector[class_index] = 1

            return (word, class_vector)

        term_freqencies = words.map(_word_tuple_to_class_vec)
        term_freqencies = term_freqencies.reduceByKey(lambda a, b: a + b)  # sum up class vectors for each word

        # compute conditional probabilities P(word | class)
        def _term_freq_to_conditional_prob(x):
            """
            Takes a (word, class_count_vector) tuple and converts it to a (word, conditional_prob_vector) tuple
            class count vector[c] is the number of time the word appears in documents of class c
            conditional_prob_vector[c] will be the conditional probability P( word | c )
            """
            LAPLACE_ESTIMATOR = 1
            word, term_freq = x
            term_freq = term_freq + LAPLACE_ESTIMATOR  # solves zero-frequency problem

            total_words = np.zeros(len(_CLASSES.value))
            for class_idx in np.arange(0, len(_CLASSES.value)):
                class_label = _CLASS_INDICES.value[class_idx]
                total_words_in_class = CLASS_WORD_COUNTS.value[class_label]
                total_words[class_idx] = total_words_in_class

            total_words = total_words + LAPLACE_ESTIMATOR*VOCAB_SIZE.value  # correct for addition of laplace estimator

            conditional_probability_vector = term_freq / total_words
            return (word, conditional_probability_vector)

        conditional_term_probabilities = term_freqencies.map(_term_freq_to_conditional_prob)
        CONDITIONAL_TERM_PROBABILITY = self.sc.broadcast(dict(conditional_term_probabilities.collect()))

        # finally, we can build the classification function:
        def _classify_document(document):
            """
            Computes argmax_k P(Y=y_k) PRODUCT_i P(x_i | Y=y_k)
            That is, this function takes the information from the training data and classifies the given document
            by finding out which class k has the highest posterior probability

            :param document: the document to classify
            :return: the label of the predicted class
            """
            words = preprocess.tokenize(document)  # split the document into words
            words = map(preprocess.remove_html_character_references, words)
            words = map(preprocess.strip_punctuation, words)
            words = map(lambda x: x.lower(), words)
            words = filter(lambda x: len(x) > 0, words)
            words = filter(lambda x: x not in _SW.value, words)

            num_classes = len(_CLASSES.value)

            posterior = np.zeros(num_classes)

            # vector of class probabilities
            marginals = [CLASS_PROBABILITY.value[class_label] for class_label in list(_CLASSES.value.keys())]

            posterior += np.log(marginals)

            # for each word, add log of word probability
            for word in words:
                if word in CONDITIONAL_TERM_PROBABILITY.value:  # if we have seen this word in training
                    conditional_prob = CONDITIONAL_TERM_PROBABILITY.value[word]
                    posterior += np.log(conditional_prob)
                # if we haven't see the word in training, we can ignore it

            class_index = np.argmax(posterior)  # index of the class with the highest posterior probability
            class_label = _CLASS_INDICES.value[class_index]
            return class_label

        self._classify_function = _classify_document
        self.has_been_trained = True

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
