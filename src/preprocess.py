"""
The preprocess module contains function definitions that are helpful in pre-processing text from example documents
"""

import string
import re


def load_stop_words(path="stopwords/generic.txt"):
    """
    Create a list of stopwords from the target text file
    Stopwords will be stripped of whitespace and converted to lowercase

    :param path: the path to the stopwords text file to load
    :return: a python list containing the stopwords from the file
    """
    with open(path, "r") as file:
        stopwords = list(file)

    return map(lambda word: word.strip().lower(), stopwords)


def remove_html_character_references(word):
    """
    This function will remove any instances of HTML character references from the given word.
    This works by replacing any substring following the pattern &...; with an empty string.

    Note that this function must be called before leading/trailing punctuation is stripped or it will NOT work

    :param word: the word from which html characters will be stripped
    :return: the same word stripped with html character references removed
    """
    # matches any string that begins with an ampersand, has one or more word characters, then ends with a semicolon
    regex = r'&\w+;'
    return re.sub(regex, "", word)


def strip_punctuation(word  , end_only = False ):
    """
    Strips punctuation from the beginning and end of the provided word

    :param word: a string containing the word to be stripped
    :param end_only: a boolean, if true only characters in the end will be considered 
    :return: the word with leading and trailing punctuation, digits, and whitespace removed
    """
    punctuation_characters = string.punctuation + string.whitespace + string.digits

    if( end_only ):
        return word.strip(punctuation_characters)
    else :
        cleaner = str.maketrans("", "", punctuation_characters)
        return word.translate(cleaner)


def tokenize(line):
    """
    Converts the given line of text into whitespace-delimited tokens

    :param line: the line of text to process
    :return: an array of tokens contained in the line of text
    """
    return line.split()


def remove_irrelevant_labels(label):
    """
    Tests if the provided label is one that we "care" about.  That is, one of the top-level "*CAT" labels
    :param label: the label to test
    :return: True if the label is a *CAT label, False otherwise
    """
    return "CAT" in label.upper()

