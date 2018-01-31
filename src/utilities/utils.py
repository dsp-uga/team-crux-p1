"""
This module contains miscellaneous utility functions
"""

import os
import logging


def custom_zip(rdd1, rdd2):
    """
    Custom zipping function for RDDs
    The native PySpark zip() function sssumes that the two RDDs have the same number of partitions and the same number
    of elements in each partition.  This zipper works without that assumption

    The results should be identical to calling rdd1.zip(rdd2)

    :param rdd1: the first rdd to zip
    :param rdd2: the second rdd to zip
    :return: a new rdd with key-value pairs where the keys are elements of the first rdd and values are
            elements of the second rdd
    """

    # create keys for join
    indexed_1 = rdd1.zipWithIndex().map(lambda x: (x[1], x[0]))
    indexed_2 = rdd2.zipWithIndex().map(lambda x: (x[1], x[0]))

    rdd_joined = indexed_1.join(indexed_2).map(lambda x: x[1])

    return rdd_joined


def print_verbose(text, log_level, threshold=0):
    """
    Conditionally prints text if the log_level meets or exceeds the provided threshold
    :param text: text to print
    :param log_level: the level of the text
    :param threshold: the minimum log level needed for this text to be printed
    :return:
    """

    # log the records
    logging.info(text)

    if log_level >= threshold:
        print(text)


def write_ouput_to_file(output, filename):
    """
    writes the output to file, if the output is supplied in hte RDD form, it'll be collected, 
    otherwise it'll be written to file  
    :param output: the list or RDD to be written to file  
    :param filename:  File name to write the file to
    :return: 
    """

    if not isinstance(output, list):
        output = output.collect()

    # ensure the ouput directory exists:
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, "w" ) as file :
        file.write( "\n".join(output) )
        file.flush()
