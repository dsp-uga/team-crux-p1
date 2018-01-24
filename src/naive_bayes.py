"""
This script is a barebones implementation of a naive bayes classifier

A
"""

from pyspark import SparkContext
import numpy as np
import preprocess # as preprocess


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


def remove_punctuation_from_list(inp):
    """
    this is a temporary function to clean list items 
    :param inp: a list of words
    :return: a cleaned list of words 
    """
    # TODO : replace this function with spark model1

    inp = [preprocess.strip_punctuation(x) for x in inp]

    ret = []
    for x in inp:
        if (len(x) > 0 and x not in ret):
            ret.append(x.lower())

    return ret



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

# duplicate each example that has multiple labels
single_label_documents = labeled_documents.flatMap(preprocess.replicate_multilabel_document)

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



#  calculate the prorir - aka the training
class_prior = class_counts.map(
    lambda x: ( x[0], calculate_prior_probability(x[1]) )
)


# TODO: prediction function


TOTAL_DOCS = labeled_documents.count()  # we'll use this to compute the prior probability of a class
TOTAL_DOCS_SINGLE_LABELED = single_label_documents.count() # we will use this count to calculate prior probability of each class since the counts are from list of docs with single label

# count how many of each class there are:
classes = single_label_documents.map(lambda x: (x[1], 1.0 /TOTAL_DOCS_SINGLE_LABELED ))  # tuple (document_class, 1 / total count) for each document, it has been devided by the total count to get rid of future devision with more ittretions
classes = classes.reduceByKey(lambda a, b: a + b)  # sum up the total number of each document class
CLASS_COUNTS = dict(classes.collect()) # this is P(Y= y_k)

print("***************",CLASS_COUNTS)

test_x = sc.textFile( "../data/X_test_vsmall.txt" ) # load the test file
test_y = sc.textFile("../data/y_test_vsmall.txt")

test_x  = test_x.map( lambda x : preprocess.tokenize(x) ) # split into words

test_x = test_x.map( lambda x : remove_punctuation_from_list(x) ) # remove punctuations from all items

words_dic = dict(  class_prior.collect() ) # this is a bad  idea! create a lookup dictionary from the class counts

test_y = test_y.collect()

test_x = test_x.collect()

hit  =0 # keep score on predictions

labels = ["CCAT", "ECAT", "GCAT", "MCAT"]
class_p = np.array(  [ CLASS_COUNTS[ x ] for x in labels ] )
for i in range( len(test_x ) ):
    doc = test_x[i]
    gt = test_y[i]

    ret =class_p # np.array(list(CLASS_COUNTS.values()))  # np.ones(4)

    for x in doc:
        xp = x.lower()
        if (xp in words_dic.keys()):
            ret = ret * (words_dic[xp] +1)

    max_index = np.argmax( ret )


    if( labels[max_index] in gt  ):
        hit +=1

    print( "Predicted : %s - GT: %s, Label Values : %s" % (  labels[max_index] ,   gt, str( ret) ))

print ( "Accuracy :  %f " % (float(hit)/len( test_x )  ))
# DANGER: don't let this line run on big datasets!!!
# this is just for testing.  We peek at the RDD to make sure it's formatted as expected
# class_counts.cache()
# class_counts.foreach(lambda x: print(x[0] + "\n" + str(x[1]) + "\n\n"))
