
from pyspark import SparkContext
import pyspark
import string

class preprocess :

    """
    this class contains the pre-processing logic for text cleaning
    
    to add a new set of stop words use the add_stopword_collection method. 
    """

    # the default set of stop words
    stopwords_path = {
        "html" : "static_data/html.txt" ,
        "generic" : "static_data/generic.txt",
        "stanford" : "static_data/stanford.txt"
    }

    stopwords_list = []
    __stop_words_loaded = False;
    spark_context = None

    def __init__(self):
        """
        class constructor, initializations go here 
        """
        self.spark_context = SparkContext.getOrCreate()


    def add_stopword_collection  ( self, title , address ):
        """
        adds a set of stopwords to be used in the preprocessing stop word removal
        :param title: the title to be used to enable/disable the filter  
        :param address: the address to load the file with the info
        :return: 
        """
        self.stopwords_path[ title ] = address


    def load_stop_words ( self, load_words  = ["html", "generic", "stanford"]    ):
        """
        loads the stop words to the list from given sources 
        :param load_words:  list of stopword sources to consider. the paths have to be already added as stopwords collecitons
        :return: 
        """
        temp = list()

        #  load the files one by one and store their words in the temp list
        for pick in load_words :
            with open( self.stopwords_path[pick] ) as file :
                temp.extend( list( file ) )


        #  remove duplicate words
        temp = list ( set( temp ))
        # remove spaces from items in the list
        self.stopwords_list=   [ pick.strip().lower() for pick in temp ]
        return  self.stopwords_list



    def remove_stop_words ( self , input_list , return_list = True   ):
        """
        this function removes stop words from an input list. it iether returns a list or an rdd
        :param input_list:  the list to look into 
        :param return_list:  if true method returns the list type, RDD type will be returned otherwise 
        :return:  the list with stop words removed. return_list can be used to change the retunrn typr 
        """

        assert isinstance( input_list , list )

        temp = self.spark_context.parallelize( input_list )
        temp = temp.filter( lambda  x : x not in self.stopwords_list.value )


        return temp.collect() if return_list else temp


    def remove_punctuations ( self, input_list , return_list  = True ):
        """
        this function removes punctuations from items in the input list / rdd
        :param input_list: the list or RDD 
        :param return_list:  chooses to return a list ot an RDD, true means list is desired
        :return: returns the list with punctuations removed
        
        """

        if isinstance( input_list , list ):
            temp= self.spark_context.parallelize( input_list )

        # define stuff to be removed
        deltab = string.punctuation + string.digits + string.whitespace  # added line
        cleaner = str.maketrans("", "", deltab)

        # remove the unwanted characters then remove empty spots
        temp = temp.map ( lambda  x:  x.translate( cleaner ) )
        temp = temp.filter ( lambda  x: len( x ) >= 1 )

        # return either list or RDD of the cleaned list
        return temp.collect() if return_list else temp


    def tokenize(self , input_rdd , stopwords) :
        """
        this method converts lines read from the initial file into tokens 
        :param input_rdd:  the input spark RDD object to process 
        :return: 
        """

        deltab = string.punctuation + string.digits + string.whitespace  # added line
        cleaner = str.maketrans("", "", deltab)
        cleaner = self.spark_context.broadcast( cleaner ).value
        # check for data type
        assert  isinstance( input_rdd , pyspark.rdd.RDD )

        temp = input_rdd.map( lambda  x: x.lower())
        # temp = temp.map( lambda x: x.encode("ASCII" , 'ignore')) # get rid of non-sacii chars
        temp = temp.map( lambda x: x.split() )
        # temp = temp.map( lambda x: self.remove_stop_words(x))
        temp = temp.filter(lambda x: x not in stopwords.value)
        temp = temp.map(lambda x:  [ y.translate(cleaner) for y in x  ])


        # temp = temp.map( lambda x: self.remove_punctuations(x) )

        return temp
