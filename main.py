"""
Driver for text classification program.
Reads command line arguments, classifies some documents, then (optionally) writes out the result
"""

from pyspark import SparkContext, SparkConf
import argparse
import os.path
import logging

from src.classifiers.EnhancedNaiveBayes import EnhancedNaiveBayesClassifier
from src.classifiers.NaiveBayes import NaiveBayesClassifier
from src.classifiers.Majority import MajorityClassifier
import src.utilities.utils as utils
import src.utilities.preprocess as preprocess

description = 'CSCI 8630 Project 1 by Team Crux.  ' \
              'This program trains a classifier using a corpus of labelled documents.  ' \
              'then classifies a set of unseen documents.  ' \
              'It is also possible to evaluate the classifier against a labelled test set.'

parser = argparse.ArgumentParser(description=description, add_help='How to use', prog='python main.py <options>')

# All args are optional
# By default, a naive bayes classifier will be trained and evaluated on the vsmall test set in the data directory
# and output the results to output/labels.txt
parser.add_argument("-d", "--dataset", default="data/X_train_vsmall.txt",
    help="Path to text file containing the documents in the training set"
         "[DEFAULT: \"data/X_train_vsmall.txt\"]")

parser.add_argument("-l", "--labels", default="data/y_train_vsmall.txt",
    help="Path to text file containing the labels for the documents in the training set"
         "[DEFAULT: \"data/y_train_vsmall.txt\"]")

parser.add_argument("-t", "--testset", default="data/X_test_vsmall.txt",
    help="Path to text file containing the documents in the testing set.  These documents will be classified"
         "[DEFAULT: \"data/X_test_vsmall.txt\"]")

parser.add_argument("-e", "--evaluate", action="store_true",
    help="Evaluate accuracy on the test set (requires file containing labels for test set)")

parser.add_argument("-m", "--testlabels", default="data/y_test_vsmall.txt",
    help="Path to text file containing the labels in the testing set (if evaluating accuracy). "
         "[DEFAULT: \"data/y_test_vsmall.txt\"]")

parser.add_argument("-s", "--stopwords", default="stopwords/all.txt",
    help="Path to the text file containing the list of stopwords. [DEFAULT: \"stopwords/all.txt/\"]")

parser.add_argument("-o", "--output", default="output",
    help="Path to the output directory where output file will be written. [DEFAULT: \"output/\"]")

parser.add_argument("-c", "--classifier", default="enb", choices=["enb", "nb", "majority"],
    help="What type of classifier to train [DEFAULT: enb]")

parser.add_argument("-v", "--verbose", action="count",
    help="Set verbosity level.  Level 0: no command-line output.  Level 1: status messages.  Level 2: Classification details.")

parser.add_argument("-lg", "--logfile", default="output/run_log.log",
    help="Path to log file [DEFAULT: output\\run_log.log]")

parser.add_argument("-df", "--dumpfile", default=None,
    help="Path to dump the word in class frequency as a CSV file [DEFAULT: None ]")


args = parser.parse_args()
if args.verbose is None:
    args.verbose = 0

# configure logger ( the file comes from input and is overwritten  )
logging.basicConfig(filename=args.logfile,
                    level=logging.INFO ,
                    filemode="w",
                    format='%(asctime)s - %(module)s.%(funcName)s - %(levelname)s :  %(message)s') # time  where it came from - LEVEL : Message



args = parser.parse_args()
if args.verbose is None:
    args.verbose = 0

conf = SparkConf().setAppName("team-crux-p1").setMaster("local") \
        .set("spark.driver.maxResultSize", "0")
sc = SparkContext(conf=conf)
# sc = SparkContext.getOrCreate()

stopwords = list(preprocess.load_stopwords(args.stopwords))
if args.classifier == "enb":
    classifier = EnhancedNaiveBayesClassifier(sc, stopwords=stopwords, dump_word_in_class_Freq=args.dumpfile)
elif args.classifier == "nb":
    classifier = NaiveBayesClassifier(sc, stopwords=stopwords)
elif args.classifier == "majority":
    classifier = MajorityClassifier()
else:
    # use default classifier
    classifier = EnhancedNaiveBayesClassifier(sc, stopwords=stopwords, dump_word_in_class_Freq=args.dumpfile)

# get the training set together:
training_docs = sc.textFile(args.dataset)  # rdd where each entry is the contents of a document
training_labels = sc.textFile(args.labels)  # each entry is a label corresponding to a document in the training set
labeled_documents = utils.custom_zip(training_docs, training_labels)  # (document_contents, label_string) tuples
labeled_documents = labeled_documents.mapValues(lambda x: x.split(','))  # label string to label array
# filter labels we don't care about
labeled_documents = labeled_documents.mapValues(preprocess.remove_irrelevant_labels)
# remove training documents with no *CAT labels
labeled_documents = labeled_documents.filter(lambda x: len(x[1]) > 0)
# duplicate each example that has multiple labels
single_label_documents = labeled_documents.flatMap(preprocess.replicate_multilabel_document)
single_label_documents.cache()

utils.print_verbose("Training classifier...", threshold=1, log_level=args.verbose)
classifier.train(single_label_documents)

# classify new examples:
test_documents = sc.textFile(args.testset)
result = classifier.classify(test_documents)

if args.evaluate:
    utils.print_verbose("Evaluation Enabled, Testing classifier using labels from %s" % args.testlabels,
                        threshold=1, log_level=args.verbose)
    test_labels = sc.textFile(args.testlabels) \
        .map(preprocess.split_by_comma) \
        .map(preprocess.remove_irrelevant_labels)

    pairs = utils.custom_zip(result, test_labels)
    correct = pairs.filter(lambda x: x[0] in x[1])
    total_test_examples = pairs.count()
    total_correct = correct.count()
    accuracy = total_correct / total_test_examples

    # print classification results if requested
    pairs.foreach(
        lambda pair: utils.print_verbose("Predicted %s, Actual: %s" % (pair[0], pair[1]),
                                         threshold=2, log_level=args.verbose )
    )

    utils.print_verbose("Accuracy on test set: %s" % accuracy, threshold=0, log_level=args.verbose)

outfile = "labels.txt"
outpath = os.path.join(args.output, outfile)
utils.print_verbose("Writing results to %s" % outpath,
                    threshold=1, log_level=args.verbose)

utils.write_ouput_to_file( result, outpath )
