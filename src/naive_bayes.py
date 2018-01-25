"""
This script is a barebones implementation of a naive bayes classifier
"""

from pyspark import SparkContext
import numpy as np
import preprocess # as preprocess
import utils


def count_words(document):
    """
    Counts the words in the provided document, excluding stop words
    """
    tokens = preprocess.tokenize(document)
    words = map(lambda x: preprocess.remove_html_character_references(x), tokens)
    words = map(lambda x: preprocess.strip_punctuation(x), words)
    words = filter(lambda x: len(x) > 0, words)
    words = list(filter(lambda x: x not in SW.value, words))

    return len(words)

def custom_zip(rdd1, rdd2):
    """
    Custom zipping function for RDDs
    The native PySpark zip() function sssumes that the two RDDs have the same number of partitions and the same number
    of elements in each partition.  This zipper works without that assumption

    The results should be identical to calling rdd1.zip(rdd2)

    :param rdd1: the first rdd to zip
    :param rdd2: the second rdd to zip
    :return: a new rdd with key-value pairs where the keys are elements of the first rdd and values are
            elements of the second rdd
    """

    # create keys for join
    indexed_1 = rdd1.zipWithIndex().map(lambda x: (x[1], x[0]))
    indexed_2 = rdd2.zipWithIndex().map(lambda x: (x[1], x[0]))

    rdd_joined = indexed_1.join(indexed_2).map(lambda x: x[1])

    return rdd_joined


def document_to_word_vec(document_tuple):
    """
    Converts a document and its label to an array of (word, class_vector) tuples
    The class vector will be a length n vector, where n is the number of classes
    The ith value of the class vector will be 1, where i is this document's class
    The indices i are determined by the `CLASSES` broadcast variable

    For example, if the document "foo bar" is passed in with class ECAT, this function will return
    [ ("foo", [0, 1, 0, 0]), ("bar", [0, 1, 0, 0]) ]
    If the document "foo bar" is passed in with class MCAT, this function will return
    [ ("foo", [0, 0, 0, 1]), ("bar", [0, 0, 0, 1]) ]

    :param document_tuple: (document, label) tuple
    :return: an array of (word, class_vector) tuples
    """
    document_contents, doc_class = document_tuple
    # extract the class index and build the count vector:
    class_index = CLASSES.value[doc_class.upper()]
    class_vector = np.zeros(len(CLASSES.value))
    class_vector[class_index] = 1

    words = preprocess.tokenize(document_contents)

    tuples = []  # holds (word, vector) tuples
    for word in words:
        tuples.append( (word, class_vector) )

    return tuples


def calculate_conditional_probability(term_frequency_vector):
    """
    Takes a word's class frequency vector and computes the conditional probabilities
    P( term | i ) for each document class i

    P( term | i) is estimated by the number of time the term occurs in class i divided by the total number of terms in i
    
    :param term_frequency_vector:  term frequency vector where position i is the term frequency for class i
    :return: an array with the conditional probability P(term | i ) for each class i
    """

    LAPLACE_ESTIMATOR = 1
    corrected_term_frequencies = term_frequency_vector + LAPLACE_ESTIMATOR

    word_counts = np.zeros(len(CLASSES.value))  # this will hold the total number of words for each class i
    for class_idx in np.arange(0, len(word_counts)):
        class_label = CLASS_INDICES.value[class_idx]
        words_in_class = LABEL_COUNTS.value[class_label]
        # correct for addition of laplace estimator:
        words_in_class += LAPLACE_ESTIMATOR * VOCAB_SIZE.value
        word_counts[class_idx] = words_in_class

    # compute marginal probabilities for each class
    return np.divide(corrected_term_frequencies, word_counts)


# TODO: we should come up with a good scheme for specifying the dataset via CL args rather than hardcoding them
TRAINING_DOCUMENTS = "../data/X_train_vsmall.txt"
TRAINING_LABELS = "../data/y_train_vsmall.txt"
TESTING_DOCUMENTS = "../data/X_test_vsmall.txt"  # presumably unlabeled data
TESTING_LABELS = "../data/y_test_vsmall.txt"
OUTPUT_FILE = "../output/y_test_vsmal.txt"

sc = SparkContext.getOrCreate()

# dictionary assigning each possible class to an integer index
classes = {
    "CCAT": 0,
    "ECAT": 1,
    "GCAT": 2,
    "MCAT": 3
}
class_indices = {
    0: "CCAT",
    1: "ECAT",
    2: "GCAT",
    3: "MCAT"
}
CLASSES = sc.broadcast(classes)
CLASS_INDICES = sc.broadcast(class_indices)

# read the stopwords files and broadcast the array of stopwords
# TODO: make the stopwords file a command-line argument
stopwords = []
stopwords.extend(preprocess.load_stopwords("stopwords/generic.txt"))
stopwords.extend(preprocess.load_stopwords("stopwords/html.txt"))
stopwords.extend(preprocess.load_stopwords("stopwords/stanford.txt"))

SW = sc.broadcast(stopwords)

file_contents = sc.textFile(TRAINING_DOCUMENTS)  # rdd where each entry is the contents of a document
document_labels = sc.textFile(TRAINING_LABELS)  # each entry is a label corresponding to a document in the training set

# join each document to its string of labels
labeled_documents = custom_zip(file_contents, document_labels)  # rdd with (document_contents, label_string) tuples

# convert comma-delimited string of labels into an array of labels
labeled_documents = labeled_documents.mapValues(lambda x: x.split(','))

# filter labels we don't care about
labeled_documents = labeled_documents.mapValues(preprocess.remove_irrelevant_labels)

# remove training documents with no *CAT labels
labeled_documents = labeled_documents.filter(lambda x: len(x[1]) > 0)

# duplicate each example that has multiple labels
single_label_documents = labeled_documents.flatMap(preprocess.replicate_multilabel_document)

single_label_documents.cache()

# find the total number of words for each label
label_counts = single_label_documents.map(lambda x: (x[1], count_words(x[0])))  \
                .reduceByKey(lambda a, b: a + b)

LABEL_COUNTS = sc.broadcast(dict(label_counts.collect()))

# convert each document to a set of words with class-count vectors
words = single_label_documents.flatMap(document_to_word_vec)
# remove html character references
words = words.map(
    lambda x: (preprocess.remove_html_character_references(x[0]), x[1])
)
# strip punctuation
words = words.map(
    lambda x: (preprocess.strip_punctuation(x[0]), x[1])
)
# to lowercase
words = words.map(
    lambda x: (x[0].lower(), x[1])
)
# filter stopwords
words = words.filter(lambda x: x[0] not in SW.value)
# make sure we don't have any empty words:
words = words.filter(lambda x: len(x[0]) > 0)

# get vocabulary
vocabulary = words.map(lambda x: x[0]).distinct()  # number of unique words in all docs
VOCAB_SIZE = sc.broadcast(vocabulary.count())

print("\n\nVOCAB SIZE: %s \n\n" % VOCAB_SIZE.value)


# sum up counts for class
class_counts = words.reduceByKey(lambda a, b: a + b)

# calculate the conditional probabilities P(x | y) for each word x
conditional_word_probabilities = class_counts.map(
    lambda x: ( x[0], calculate_conditional_probability(x[1]) )
)
CONDITIONAL_WORD_PROBABILITIES = sc.broadcast(dict(conditional_word_probabilities.collect()))

TOTAL_DOCS = single_label_documents.count()

# count how many of each class there are:
classes = single_label_documents.map(lambda x: (x[1], 1))   # (class, 1) tuples
classes = classes.reduceByKey(lambda a, b: a + b)  # sum up the total number of each document class
marginal_class_probs = classes.map(lambda x: (x[0], x[1]/TOTAL_DOCS))  # convert from counts to marginal probabilities
MARGINAL_CLASS_PROBS = sc.broadcast(   # this holds P(Y=k) for each class k
    dict(marginal_class_probs.collect())
)


# we now have all information needed to classify an unseen document:
def classify(document):
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
    words = filter(lambda x: x not in SW.value, words)

    num_classes = len(CLASSES.value.keys())
    marginal = np.zeros(num_classes)

    # start with the marginal probability of each class
    for class_label, marginal_probability in MARGINAL_CLASS_PROBS.value.items():
        index = CLASSES.value[class_label]  # numeric index for this class
        marginal[index] = np.log(marginal_probability)

    # multiply the marginal probability of class y with the conditional probability of each word
    posterior = marginal
    for word in words:
        if word in CONDITIONAL_WORD_PROBABILITIES.value:
            conditional = CONDITIONAL_WORD_PROBABILITIES.value[word]
        else:
            # if the word has never been seen before
            # TODO: need someone to check if this is the right thing to do
            conditional = np.ones(num_classes) * 1/num_classes

        posterior = posterior + np.log(conditional)

    max_index = np.argmax(posterior)  # index of the class with the highest posterior probability
    class_label = CLASS_INDICES.value[max_index]

    return class_label


test_documents = sc.textFile(TESTING_DOCUMENTS)  # read in test data
# the following makes a fairly big assumption that the test documents will be kept in order
result = test_documents.map(lambda x: classify(x))

test_labels = sc.textFile(TESTING_LABELS)\
    .map(preprocess.split_by_comma)\
    .map(preprocess.remove_irrelevant_labels)

pairs = custom_zip(result, test_labels)

correct = pairs.filter(lambda x: x[0] in x[1])

total_test_examples = pairs.count()
total_correct = correct.count()

accuracy = total_correct / total_test_examples

pairs.foreach(lambda pair: print("Predicted %s, Actual: %s" % (pair[0], pair[1]) ))

print("Estimated accuracy: %s" % accuracy)

# TODO need to optionally output the results to a file
# utils.save_to_file( result, OUTPUT_FILE )
