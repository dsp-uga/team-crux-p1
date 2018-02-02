"""
Preprocessing module
Contains functions which are helpful for preprocessing text
"""

import string
import re


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


def strip_punctuation(word, end_only=False):
    """
    Strips punctuation from the beginning and end of the provided word

    :param word: a string containing the word to be stripped
    :param end_only: a boolean, if true only characters in the end will be considered 
    :return: the word with leading and trailing punctuation, digits, and whitespace removed
    """
    punctuation_characters = string.punctuation + string.whitespace + string.digits

    if end_only:
        return word.strip(punctuation_characters)
    else:
        cleaner = str.maketrans("", "", punctuation_characters)
        return word.translate(cleaner)


def tokenize(line):
    """
    Converts the given line of text into whitespace-delimited tokens

    :param line: the line of text to process
    :return: an array of tokens contained in the line of text
    """
    return line.split()


def split_by_comma(line):
    """
    Converts the given line of text into comma-delimited tokens

    :param line: the line of text to process
    :return: an array of tokens contained in the line of text
    """
    return line.split(",")


def remove_irrelevant_labels(labels):
    """
    Filters an array of labels, removing non "CAT" labels

    :param labels: array of labels
    :return: an array of labels containing only the top-level "CAT" labels
    """
    filtered = filter(lambda x: "CAT" in x.upper(), labels)
    return list(filtered)


def replicate_multilabel_document(document_tuple):
    """
    Takes a document and a set of labels and returns multiple copies of that document - each with a single label
    For example, if we pass ('foo', ['a', 'b']) to this function, then it will return
        [ ('foo', 'a'), ('foo', 'b') ]

    :param document_tuple: (document, array_of_labels)
    :return: array of (document, label) pairs
    """
    document_contents, labels = document_tuple

    single_label_documents = []
    for label in labels:
        single_label_documents.append((document_contents, label))

    return single_label_documents

