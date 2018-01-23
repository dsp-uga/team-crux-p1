"""
This script is a barebones implementation of a naive bayes classifier

A
"""

from pyspark import SparkContext
import numpy as np
import src.preprocess as preprocess


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


def add_one_to_aray(class_count_array):
    # TODO: we may be better off doing this in the prediction function itself.
    """
    Adds one to the counts to get rid of the multiplied by zero problem
    
    :param document_count_array: the input list to add ones to  
    :return: an array with same structure as input, all items 
    """
    return class_count_array + 1


def calculate_prior_probability(class_word_frequency):
    """
    Takes a word's class frequency vector and computes the prior probabilities
    P( word | y ) for each document class y
    
    In Naive Bayes, P(x|yk) = Freq(x) / Sum_k ( Freq(x) )
    if input is [1,1,1,1] output will be [1/4,1/4,1/4,1/4]
    if input is [3,1,2,1] output will be [3/7, 1/7, 2/7, 1/7]
    
    :param class_word_frequency:  term frequency vector where position i is the term frequency for class i
    :return: an array with prior probabilities for each class
    """

    # calculate the total occurrences of this word in all classes
    sum = np.sum( class_word_frequency )

    # for each class y, calculate the probability of observing the word in that class
    return class_word_frequency / sum


# TODO: we should come up with a good scheme for specifying the dataset via CL args rather than hardcoding them
TRAINING_DOCUMENTS = "../data/X_train_vsmall.txt"
TRAINING_LABELS = "../data/y_train_vsmall.txt"
TESTING_DOCUMENTS = "../data/X_test_vsmall.txt"  # presumably unlabeled data

sc = SparkContext.getOrCreate()

# dictionary assigning each possible class to an integer index
classes = {
    "CCAT": 0,
    "ECAT": 1,
    "GCAT": 2,
    "MCAT": 3
}
CLASSES = sc.broadcast(classes)

# read the stopwords files and broadcast the array of stopwords
# TODO: make the stopwords file a command-line argument
stopwords = []
stopwords.extend(preprocess.load_stop_words("stopwords/generic.txt"))
stopwords.extend(preprocess.load_stop_words("stopwords/html.txt"))
stopwords.extend(preprocess.load_stop_words("stopwords/stanford.txt"))

SW = sc.broadcast(stopwords)

file_contents = sc.textFile(TRAINING_DOCUMENTS)  # rdd where each entry is the contents of a document
document_labels = sc.textFile(TRAINING_LABELS)  # each entry is a label corresponding to a document in the training set

# join each document to its string of labels
labeled_documents = file_contents.zip(document_labels)  # rdd with (document_contents, label_string) tuples

# convert comma-delimited string of labels into an array of labels
labeled_documents = labeled_documents.mapValues(lambda x: x.split(','))

# filter labels we don't care about
labeled_documents = labeled_documents.mapValues(preprocess.remove_irrelevant_labels)

# remove training documents with no *CAT labels
labeled_documents = labeled_documents.filter(lambda x: len(x[1]) > 0)


TOTAL_DOCS = labeled_documents.count()  # we'll use this to compute the prior probability of a class

# duplicate each example that has multiple labels
single_label_documents = labeled_documents.flatMap(preprocess.replicate_multilabel_document)

# TODO we can use the single_label_documents RDD to count how many of each class there are

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

# sum up counts
class_counts = words.reduceByKey(lambda a, b: a + b)

# TODO: class_counts holds how many times each word appears per class.  This should be all we need to compute prior probs and do classification!

#  calculate the proriri - aka the training
class_priori = class_counts.map(
    lambda x: ( x[0], calculate_prior_probability(x[1]) )
)


# TODO: prediction function






# DANGER: don't let this line run on big datasets!!!
# this is just for testing.  We peek at the RDD to make sure it's formatted as expected
class_counts.cache()
class_counts.foreach(lambda x: print(x[0] + "\n" + str(x[1]) + "\n\n"))
