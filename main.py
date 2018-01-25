"""
Driver for text classification program.
Reads command line arguments, classifies some documents, then (optionally) writes out the result
"""
# TODO: make this file actually do what it's supposed to do^^

from pyspark import SparkContext
from src.classifiers.NaiveBayes import NaiveBayesClassifier
import src.utilities.utils as utils
import src.utilities.preprocess as preprocess

TRAINING_DOCUMENTS = "data/X_train_vsmall.txt"
TRAINING_LABELS = "data/y_train_vsmall.txt"
TESTING_DOCUMENTS = "data/X_test_vsmall.txt"  # presumably unlabeled data
TESTING_LABELS = "data/y_test_vsmall.txt"
OUTPUT_FILE = "output/y_test_vsmal.txt"

sc = SparkContext.getOrCreate()
stopwords = []
stopwords.extend(preprocess.load_stopwords("stopwords/generic.txt"))
stopwords.extend(preprocess.load_stopwords("stopwords/html.txt"))
stopwords.extend(preprocess.load_stopwords("stopwords/stanford.txt"))

classifier = NaiveBayesClassifier(sc, stopwords=stopwords)

training_docs = sc.textFile(TRAINING_DOCUMENTS)  # rdd where each entry is the contents of a document
training_labels = sc.textFile(TRAINING_LABELS)  # each entry is a label corresponding to a document in the training set
labeled_documents = utils.custom_zip(training_docs, training_labels)  # (document_contents, label_string) tuples
labeled_documents = labeled_documents.mapValues(lambda x: x.split(','))  # label string to label array
# filter labels we don't care about
labeled_documents = labeled_documents.mapValues(preprocess.remove_irrelevant_labels)
# remove training documents with no *CAT labels
labeled_documents = labeled_documents.filter(lambda x: len(x[1]) > 0)
# duplicate each example that has multiple labels
single_label_documents = labeled_documents.flatMap(preprocess.replicate_multilabel_document)
single_label_documents.cache()

classifier.train(single_label_documents)

# test the classifier
test_documents = sc.textFile(TESTING_DOCUMENTS)  # read in test data
result = classifier.classify(test_documents)

test_labels = sc.textFile(TESTING_LABELS)\
    .map(preprocess.split_by_comma)\
    .map(preprocess.remove_irrelevant_labels)

pairs = utils.custom_zip(result, test_labels)
correct = pairs.filter(lambda x: x[0] in x[1])
total_test_examples = pairs.count()
total_correct = correct.count()
accuracy = total_correct / total_test_examples

pairs.foreach(lambda pair: print("Predicted %s, Actual: %s" % (pair[0], pair[1]) ))

print("Estimated accuracy: %s" % accuracy)

# TODO: optionally output result to file

