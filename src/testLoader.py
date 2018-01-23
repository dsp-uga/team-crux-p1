"""
Tests if the preprocessing functions seem to be working as planned
"""

import src.preprocess as preprocess
from pyspark import SparkContext

sc = SparkContext.getOrCreate()

# read the stopwords files and broadcast the array of stopwords
stopwords = []
stopwords.extend(preprocess.load_stop_words("stopwords/generic.txt"))
stopwords.extend(preprocess.load_stop_words("stopwords/html.txt"))
stopwords.extend(preprocess.load_stop_words("stopwords/stanford.txt"))

SW = sc.broadcast(stopwords)

# read each line of the file into an RDD
file_contents = sc.textFile("../data/X_train_vsmall.txt")

words = file_contents.flatMap(preprocess.tokenize)  # tokenize the line of text

words = words.map(preprocess.remove_html_character_references)  # remove html character encodings

words = words.map(preprocess.strip_punctuation)  # remove leading/trailing punctuation

words = words.map(lambda word: word.lower())  # lowercase words

words = words.filter(lambda word: word not in SW.value)  # remove stopwords

words = words.filter(lambda word: len(word) > 1)  # remove words that are one character or less

print(words.collect())
