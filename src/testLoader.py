from preprocess import preprocess
from pyspark import SparkContext

sc  = SparkContext.getOrCreate()

file  = sc.textFile( "../data/X_train_vsmall.txt" )


p = preprocess()
sl =sc.broadcast(  p.load_stop_words())

print (sl)
f2 = p.tokenize(file , sl)


print (f2.collect())
