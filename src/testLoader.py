from .preprocess import Preprocess
from pyspark import SparkContext

sc  = SparkContext.getOrCreate()

file  = sc.textFile( "../data/X_train_vsmall.txt" )


p = Preprocess()
sl =sc.broadcast(  p.load_stop_words())

print (sl)
f2 = p.tokenize(file , sl)


print (f2.collect())
