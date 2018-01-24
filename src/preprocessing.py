import preprocess
import numpy as np
import glob
import string


html = [line.strip() for line in open("stopwords/html.txt", "r")]
generic = [line.strip() for line in open("stopwords/generic.txt", "r")]
stanford = [line.strip() for line in open("stopwords/stanford.txt", "r")]

# reads the document file and writes the preprocessed document file into project/test_preprocess/
for file in glob.glob("../data/X_preprocess_test.*"):
	infile = open(file, "r")
	outfile = open("../test_preprocess/"+file[8:], "w")
	lines = infile.readlines()
	for line in lines:
		pline = ' '.join(filter(lambda x: x.lower() not in html, line.split()))
		pline = pline.translate(None, string.punctuation).translate(None, string.digits).lower()
		pline = ' '.join(filter(lambda x: x.lower() not in generic, pline.split()))
		pline = ' '.join(filter(lambda x: x.lower() not in stanford, pline.split()))
		pline += '\n'
		outfile.write(pline)

# reads the labels file and writes the preprocessed labels file into project/test_preprocess/
for file in glob.glob("../data/y_preprocess_test.*"):
	infile = open(file, "r")
	outfile = open("../test_preprocess/"+file[8:], "w")
	lines = infile.readlines()
	for line in lines:
		pline = line.strip()
		pline = ' '.join(filter(lambda x: "CAT" in x.upper(), pline.split(',')))
		pline += '\n'
		outfile.write(pline)