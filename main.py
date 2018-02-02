"""
Driver file that collects command line arguments and calls the main method from the src package
"""
import logging
import argparse
import src.__main__

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

parser.add_argument("-c", "--classifier", default="enb", choices=["enb", "nb", "majority", "css"],
    help="What type of classifier to train [DEFAULT: enb]")

parser.add_argument("-v", "--verbose", action="count",
    help="Set verbosity level.  Level 0: no command-line output.  Level 1: status messages.  Level 2: Classification details.")

parser.add_argument("-df", "--dumpfile", default=None,
    help="Path to dump the word in class frequency as a CSV file [DEFAULT: None ]")


args = parser.parse_args()
if args.verbose is None:
    args.verbose = 0

src.__main__.main(args)
