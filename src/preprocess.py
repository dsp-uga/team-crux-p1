
# from pyspark  import SparkContext



class preprocess :

    """
    this class contains the pre-processing logic for text cleaning
    """

    # the default set of stop words 
    stopwords_path = {
        "html" : "static_data/html.txt" ,
        "generic" : "static_data/generic.txt",
        "stanford" : "static_data/stanford.txt"
    }

    stopwords_list = []

    def add_stopword_collection  ( self, title , address ):
        """
        adds a set of stopwords to be used in the preprocessing stop word removal
        :param title: the title to be used to enable/disable the filter  
        :param address: the address to load the file with the info
        :return: 
        """
        self.stopwords_path[ title ] = address


    def load_words ( self, load_words  = ["html", "generic", "stanford"]    ):
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
        self.stopwords_list= [ pick.strip() for pick in temp ]

