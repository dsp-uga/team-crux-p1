"""
Driver for text classification program.
Reads command line arguments, classifies some documents, then (optionally) writes out the result
"""
# TODO: make this file actually do what it's  supposed to do^^


from pyspark import SparkContext
from src.classifiers.NaiveBayes import NaiveBayesClassifier
import src.utils as utils
import src.preprocess as preprocess

TRAINING_DOCUMENTS = "data/X_train_vsmall.txt"
TRAINING_LABELS = "data/y_train_vsmall.txt"
TESTING_DOCUMENTS = "data/X_test_vsmall.txt"  # presumably unlabeled data
TESTING_LABELS = "data/y_test_vsmall.txt"
OUTPUT_FILE = "output/y_test_vsmal.txt"

sc = SparkContext.getOrCreate()
stopwords = []
stopwords.extend(preprocess.load_stopwords("src/stopwords/generic.txt"))
stopwords.extend(preprocess.load_stopwords("src/stopwords/html.txt"))
stopwords.extend(preprocess.load_stopwords("src/stopwords/stanford.txt"))

classifier = NaiveBayesClassifier(sc, stopwords=stopwords)

file_contents = sc.textFile(TRAINING_DOCUMENTS)  # rdd where each entry is the contents of a document
document_labels = sc.textFile(TRAINING_LABELS)  # each entry is a label corresponding to a document in the training set

# join each document to its string of labels
labeled_documents = utils.custom_zip(file_contents, document_labels)  # rdd with (document_contents, label_string) tuples

# convert comma-delimited string of labels into an array of labels
labeled_documents = labeled_documents.mapValues(lambda x: x.split(','))

# filter labels we don't care about
labeled_documents = labeled_documents.mapValues(preprocess.remove_irrelevant_labels)

# remove training documents with no *CAT labels
labeled_documents = labeled_documents.filter(lambda x: len(x[1]) > 0)

# duplicate each example that has multiple labels
single_label_documents = labeled_documents.flatMap(preprocess.replicate_multilabel_document)

single_label_documents.cache()
classifier.train(single_label_documents)

