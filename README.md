# team-crux-p1

Distributed document classification using Apache Spark.

This project seeks to build a model capable of classifying news stories into one of the following four categories:
* Corporate/Industrial (CCAT)
* Economics (ECAT)
* Government/Social (GCAT)
* Markets (MCAT)

This project is capable of building a few different classifiers, including:
* Majority Classifier
* Basic Naive Bayes Classifier
* Enhanced Naive Bayes Classifier

The basic naive bayes classifier is a standard implementation of naive bayes for document classification.
It generally seems to exhibit decent performance for small-medium sized data sets but exhibits poor performance
on large datasets.

The enhanced naive bayes classifier includes several improvements:
* feature selection that removes terms with similar frequency across all four classes
* term-frequency inverse-class-frequency (TF-ICF) weighting of words
* Various performance improvements

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes.

### Prerequisites

This project uses [Apache Spark](https://spark.apache.org).  You'll need to have Spark installed on the target cluster.  
The ```SPARK_HOME``` environment variable should be set, and the Spark binaries should be in your system path.

Dependencies are managed using the [Conda](https://conda.io/docs/) package manager.  You'll need to install Conda to get setup.

### Installing Dependencies

The environment.yml file is used by Conda to create a virtual environment that includes all the project's dependencies (including Python!)

Navigate to the project directory and run the following command

```
conda env create
```

This will create a virtual environment named "team-crux-p1".  Activate the virtual environment with the following command

```
source activate team-crux-p1
```

After the environment has been activated, the program can be run as follows:

```
python main.py <options>
```

### Usage

One very small example dataset is included in the ```data``` directory.  
When run with default options, the program will train an enhanced naive bayes classifier on the example
dataset and write the results to ```output/labels.txt```.

Run ```python main.py -h``` to view a short synopsis of the available options.  
A detailed description of each is provided below.

Options:
* ```-d, --dataset <path/to/training/documents.txt>```

    Path to text file containing the documents in the training set.  Each document should be on a separate line.
    [DEFAULT: "data/X_train_vsmall.txt"]

* ```-l, --labels <path/to/training/labels.txt>```
    Path to text file containing the labels for the documents in the training set.  Each label should
    occupy a new line and should correspond to the documents in the training set.
    [DEFAULT: "data/y_train_vsmall.txt"]
                        
* ```-t, --testset <path/to/test/data.txt>```
    Path to text file containing the documents in the testing set. 
    The classifier built using the training set will be used to classify these documents.
    This file should follow the same format as the training dataset.
    [DEFAULT: "data/X_test_vsmall.txt"]
                        
* ```-e, --evaluate```        
    If this flag is set, then the labels output for the test set will be compared against 
    the provided set of test labels and the accuracy will be output to the console.
    If this flag is set, then a set of test labels MUST be provided
    
* ```-m, --testlabels <path/to/test/labels.txt>```
    Path to text file containing the labels in the testing set (if evaluating accuracy). 
    This option is ignored if the ```evaluate``` flag is not set
    [DEFAULT: "data/y_test_vsmall.txt"]
    
* ```-s, --stopwords <path/to/stopwords.txt>```
    Path to the text file containing the list of stopwords (if using custom list).
    The repository includes a small list of common stopwords sourced from 
    [this repository](https://code.google.com/archive/p/stop-words/)
    [DEFAULT: "stopwords/all.txt/"]
    
* ```-o, --output <outpath/path/>```
    Path to the output directory where output file will be
    written.  After classifying the test set, the labels will be written a file called part-00000 in
    this directory.
    [DEFAULT: "output/"]
                        
* ```-c, --classifier {enb, nb, majority}```
    What type of classifier to train. 
     ```enb``` = Enhanced naive bayes.  ```nb``` = basic Naive Bayes.
    [DEFAULT: "enb"]    
                        
* ```-v, --verbose```         
    Set verbosity level.  Each additional ```-v``` raises the verbosity level by 1.
    Level 0: no command-line output.
    Level 1: status messages. 
    Level 2: Classification details.
    

## Running the tests

This project uses Python's built-in unittest module for running tests.
Tests are located in the `.tests` package and currently cover the reusable functons in the src.utilities module.

Run tests by running the following (with your conda env activated) in the project directory:
```python -m unittest discover```


## Deployment

This repository includes a shell script [submit.sh](submit.sh) that will package the source files into an 
[egg](http://peak.telecommunity.com/DevCenter/EasyInstall)
and submit the pyspark job to a Google Cloud Compute cluster.  It assumes that the 
[Google Cloud SDK](https://cloud.google.com/sdk/) is installed and on the system path.  
You will also need to have authenticated with the gcloud SDK using the google account linked to the target cluster.

The [submit.sh](submit.sh) file requires minor custom configuration.  
The name of the target cluster should be changed to your own cluster name.
The other program options work as expected and may be changed as desired.  

To use the submission script, first activate the conda environment for the project (see Installation section for details).
Then you can simply:
``` ./submit.sh ```

Thanks to [Chris Barrick](https://github.com/cbarrick) for his help with creating the 
`setup.py` and `submit.sh` scripts!

## Built With

* [Python 3.6](https://www.python.org/)
* [Apache Spark](https://spark.apache.org)
* [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html) - Python API for [Apache Spark](https://spark.apache.org/)
* [Conda](https://conda.io/docs/) - Package Manager

## Contributing

There are no specific guidelines for contributing.  Feel free to send a pull request if you have
an improvement.

## Versioning

We use the [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) 
workflow to organize releases and development of new features.

## Authors

See the [contributors](https://github.com/dsp-uga/team-crux-p1/contributors.md) file for details

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments

* This project was completed as a part of the Data Science Practicum 2018 course at the University of Georgia
* [Dr. Shannon Quinn](https://github.com/magsol)
 is responsible for the problem formulation and initial guidance towards solution methods.  He also 
 provided the very small data set included in this repository
* A. Balucha for his [repository of stopwords](https://code.google.com/archive/p/stop-words/)
* [Chris Barrick](https://github.com/cbarrick) for providing the setup.py and submit.sh scrips

