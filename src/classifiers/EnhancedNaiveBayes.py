import pyspark
import numpy as np
import csv
from .Classifier import Classifier
import src.utilities.preprocess as preprocess
from nltk.stem import WordNetLemmatizer


class EnhancedNaiveBayesClassifier(Classifier):
    def __init__(self, spark_context, stopwords=[], dump_word_in_class_Freq=None):
        """
        This classifier uses a modified version of Naive Bayes to classify documents from the Reuters Corpus
        It includes enhancements:
            * term filtering based on variance among classes
            * term weighting by tf-icf scores
        See project wiki for details

        This classifier assumes that the program is running in an Apache Spark cluster
        :param spark_context: the Spark context object in which the RDD operations are being performed
        :param stopwords: file containing stopwords to be removed from the corpus
        :param dump_word_in_class_Freq : an optional variable to set for the word in lcass frequency to be stored to
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

        # set the path for the file which will contain the dump of word for class frequency
        self.dump_word_in_class_Freq = dump_word_in_class_Freq

        # mapping from numeric index back to class label
        self.CLASS_INDICES = self.sc.broadcast({v: k for k, v in self.CLASSES.value.items()})

        # list of stopwords:
        self.SW = self.sc.broadcast(stopwords)

        self.lemmatizer = WordNetLemmatizer()

    def train(self, data):
        """
        Trains the classifier using the specified set of labeled documents
        Builds a private function to classify a single document

        :param data: RDD containing ( document_contents, document_label ) pairs
        :return: None
        """
        assert (isinstance(data, pyspark.RDD))
        _TOTAL_DOCS = data.count()
        TOTAL_DOCS = self.sc.broadcast(data.count())
        # get the number of documents per class
        class_unit_counts = data.map(lambda x: (x[1], 1))  # (class_label, 1) tuples
        class_counts = class_unit_counts.reduceByKey(lambda a, b: a + b)

        # we can now calculate class probabilities
        class_probabilities = class_counts.mapValues(lambda x: x / TOTAL_DOCS.value)
        CLASS_PROBABILITY = self.sc.broadcast(dict(class_probabilities.collect()))

        # tokenize and preprocess documents:
        _SW = self.SW
        _lemmatizer = self.lemmatizer
        processed = data.map(lambda tuple: (tuple[1], tuple[0]))  # temporarily swap to (class, document_contents)
        words = processed.flatMapValues(preprocess.tokenize)  # (class, word) pairs
        words = words.mapValues(preprocess.remove_html_character_references)
        words = words.mapValues(preprocess.strip_punctuation)
        words = words.mapValues(lambda x: x.lower())
        words = words.mapValues(lambda x: _lemmatizer.lemmatize(x))  # lemmatization from NLTK
        words = words.filter(lambda x: len(x[1]) > 0)  # make sure we don't include empty string
        words = words.filter(lambda x: x[1] not in _SW.value)  # remove stopwords from corpus
        words.cache()

        # extract the vocabulary size (number of unique words in the corpus)
        vocab = words.map(lambda x: x[1]).distinct()
        VOCAB_SIZE = self.sc.broadcast(vocab.count())

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

        def _tf_icf(x):
            """
            Takes a (word, class_vector) tuple and calculates tf/icf scores of the word for each class
            tf = log(1 + f(i,j)), where f(i,j) is number of occurrences of word i in class j
            icf = log((1 + N)/(1 + n(j))),
            where N is number of documents in corpus, n(j) = number of times term j occurred in corpus

            tficf score = tf * icf
            """
            term = x[0]
            frequencies = x[1] + 1
            tf = np.log(frequencies)
            nj = np.sum(frequencies)  # number of documents in which the word appears
            icf = np.log((1 + _TOTAL_DOCS) / (1 + nj))
            tficf = tf * icf
            return (term, tficf)

        tficf_scores = term_freqencies.map(_tf_icf)

        tficf_scores = tficf_scores.reduceByKey(lambda a, b: a + b)  # sum up class vectors for each word

        # TODO: we probably shouldn't keep this around forever.  Not a feature that end-users need
        if (self.dump_word_in_class_Freq != None):
            with open(self.dump_word_in_class_Freq, "w") as word_output_file:
                writer = csv.writer(word_output_file)
                writer.writerows(tficf_scores.map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3])).collect())

        def filter_by_std_deviation(arr):
            """
            Checks that the standard deviation of elements in an array is above a certain threshold

            :param arr: A NumPy Array
            :return: True if the standard deviation is above 1.8 or if one of the elements is zero, False otherwise
            """
            THRESHOLD = 0  # magic number chosen by running some statistical procedures against a large dataset
            return np.std(arr) !=  THRESHOLD #or np.sum(arr) < 4

        # find words with equal frequency in all classes
        # Note: this is a form of feature selection - we remove meaningless features - See project wiki for more info
        # TODO: sections should be added to wiki
        tficf_scores = tficf_scores.filter(lambda x: filter_by_std_deviation(x[1]))

        # count all words
        _TOTAL_WORDS = term_freqencies.map(lambda x: x[1]).reduce(lambda a, b: a + b)

        _TOTAL_WORDS = self.sc.broadcast(_TOTAL_WORDS)

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

            total_words = _TOTAL_WORDS.value + LAPLACE_ESTIMATOR * VOCAB_SIZE.value  # correct for addition of laplace estimator

            conditional_probability_vector = term_freq / total_words
            return (word, conditional_probability_vector)

        conditional_term_probabilities = tficf_scores.map(_term_freq_to_conditional_prob)
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
            num_classes = len(_CLASSES.value)

            posterior = np.zeros(num_classes)

            # vector of class probabilities
            marginals = [CLASS_PROBABILITY.value[class_label] for class_label in list(_CLASSES.value.keys())]

            posterior += np.log(marginals)

            # for each word, add log of word probability
            tokens = preprocess.tokenize(document)
            for token in tokens:
                word = preprocess.remove_html_character_references(token)
                word = preprocess.strip_punctuation(word).lower()
                if word not in _SW.value and len(word) > 0:  # then it's actually a word we want to consider
                    # if we have seen this word in training
                    if word in CONDITIONAL_TERM_PROBABILITY.value:
                        conditional_prob = CONDITIONAL_TERM_PROBABILITY.value[word]
                        posterior += np.log(conditional_prob)
                        # if we haven't see the word in training, we can ignore it

            class_index = np.argmax(posterior)  # index of the class with the highest posterior probability
            class_label = _CLASS_INDICES.value[class_index]
            return class_label

        self._classify_function = _classify_document
        self.has_been_trained = True
