
## this File does some basic analysis on the word in class frequncy 

library("matrixStats")
library("dplyr")

## remove variables from memory 
rm( list = ls() )


## set working directory - just to make sure
setwd( "/home/omido/Courses/8360/team-crux-p1/analysis/")

## load the CSV file
word.class.df <- read.table( "freqdump.csv"  , sep = "," , header = TRUE)


## Print the summary of the variables : 
summary( word.class.df)



# overlaying histograms of counts
hist(   word.class.df$ECAT,5, col = rgb( 0.4,0,0.3,0.5 )   )
hist(   word.class.df$MCAT, 5, col = rgb( 0,1,0,0.5 ) , add=TRUE)
hist(   word.class.df$GCAT, 5, col = rgb( 0,0,1,0.5 ),  add=TRUE)
hist(   word.class.df$CCAT, 5, col = rgb( 0.1,0.3,0.2,0.5 ) , add=TRUE)
box()

## well it looked ugly :(  

# creater a matrix of the numbers and then transpose it to make calculations eaiser 
theMat  <-  as.matrix(word.class.df[ c( "MCAT", "CCAT", "GCAT", "ECAT" ) ]);
theMat.t <- t( theMat )

#calculate mean, sum  and SD 
sds<- colSds(theMat.t )  
avgs <- colMeans(theMat.t  )
sums <- colSums( theMat.t )
mins <- colMins( theMat.t )
vars <- colVars( theMat.t)
maxs <- colMaxs(theMat.t)

# save calculated values back to dataframe 
word.class.df$sds  <-  sds  
word.class.df$sums <- sums
word.class.df$avgs <- avgs
word.class.df$mins <- mins
word.class.df$maxes <- maxs
word.class.df$vars <- vars

# now check how the sd looks like
hist ( sds, n =10 )

# filter the rows to ones with less standard deviation 
words_with_small_sd <-filter( word.class.df , sds<1.5 )

# take a peek at the new data
head ( words_with_small_sd , n = 10 )

# plot the standard deviations histogram 
hist( words_with_small_sd$sds , 150 , col = blues9  , main="Hist - per class frequency standard deviation for words with sd<1.5" , xlab = "Standard Deviation"  )

# plot standard deviation CDF
cdf_p = ecdf( words_with_small_sd$sds )
plot( cdf_p , main="CDF - standard Devations" , ylab = "CDF", xlab ='Standard Deviation') 

# plot sums against 
plot ( word.class.df$sds ~ word.class.df$sums )

# plot fields against each other
plot(  word.class.df[ c( "MCAT", "CCAT", "GCAT", "ECAT" , "sds" , "sums", "avgs" ) ]  )

write.csv( word.class.df , "words.csv" )
