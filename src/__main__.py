"""
Driver for text classification program.
"""

from pyspark import SparkContext, SparkConf
import os.path

from src.classifiers.EnhancedNaiveBayes import EnhancedNaiveBayesClassifier
from src.classifiers.NaiveBayes import NaiveBayesClassifier
from src.classifiers.Majority import MajorityClassifier
import src.utilities.utils as utils
import src.utilities.preprocess as preprocess


def main(args):
    """ args will be a dictionary of command-line arguments as parsed by argparser"""

    configuration = SparkConf().setAppName("team-crux-p1")\
                    .set('spark.hadoop.validateOutputSpecs', "false") \
                    .set("spark.driver.maxResultSize", "3g") \
                    .set("spark.driver.cores", "2") \

    sc = SparkContext.getOrCreate(configuration)

    stopwords = sc.textFile(args.stopwords)  # rdd where each entry is a stopword
    stopwords = stopwords.map(lambda x: x.strip().lower()).collect()
    stopwords = list(stopwords)

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

    # output result to text file
    utils.print_verbose("Writing results to %s" % args.output,
                        threshold=1, log_level=args.verbose)

    result.coalesce(1).saveAsTextFile(args.output)
