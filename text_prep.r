### TODO
### convert binary presence features to tfidf?
### tag bert on as well

library(stringr)
library(glue)
library(dplyr)
library(fastDummies)

# set working directory to parent of fpc_files
#setwd(".")

num_files_input <- 5
num_files_input <- 19

# read in some sample data
path = "fpc_files/fpc_{i}_with_code.csv"

fpc1 <- data.frame()
for (i in 1:num_files_input) {
  fpc1 <- rbind(fpc1, read.csv(glue(path)))
}

# pull just the data we want
fpc1$new_relations <- as.factor(fpc1$relation)
fpc1$new_word <- as.factor(fpc1$word)
fpc1$new_pos <- as.factor(fpc1$pos)
fpc1$new_source <- as.factor(fpc1$word_source)

fpc1$CodeType <- as.factor(fpc1$CodeType)

fpc1$id <- NULL
fpc1$sid <- NULL
fpc1$index <- 0
for (i in 1:nrow(fpc1)) {
  fpc1[i, "index"] <- i-1
}
sample <- data.frame(fpc1[, "new_word"])

# create lag variables
lagjunk <- fpc1 
lagjunk$tid <- fpc1$tid+1
lagjunk$index <- fpc1$index+1
colnames(lagjunk)[2:(ncol(lagjunk)-1)] <- paste0("before_", colnames(lagjunk)[2:(ncol(lagjunk)-1)])
junk <- merge(fpc1, lagjunk, by=c("index", "tid"), all.x=T)

# create lead variables
leadjunk <- fpc1 
leadjunk$tid <- fpc1$tid-1
leadjunk$index <- fpc1$index-1
colnames(leadjunk)[2:(ncol(lagjunk)-1)] <- paste0("after_", colnames(leadjunk)[2:(ncol(lagjunk)-1)])
junk <- merge(junk, leadjunk, by=c("index", "tid"), all.x=T)

# create a series of dummy variable
junk <- junk[,which(colnames(junk) %in% c("CodeType", "tid", "new_relations", "new_word", "new_pos", "new_source", "before_new_relations", "before_new_word", "before_new_pos", "before_new_source", "after_new_relations", "after_new_word", "after_new_pos", "after_new_source", "sentiment", "before_sentiment", "after_sentiment"))]
junk$new_word <- as.numeric(junk$new_word)
junk$new_source <- as.numeric(junk$new_source)
junk$before_new_word <- as.numeric(junk$before_new_word)
junk$before_new_source <- as.numeric(junk$before_new_source)
junk$after_new_word <- as.numeric(junk$after_new_word)
junk$after_new_source <- as.numeric(junk$after_new_source)

junk <- dummy_cols(junk, select_columns = c("new_word", "before_new_word", "after_new_word", "new_relations", "new_pos", "before_new_relations", "before_new_pos", "after_new_relations", "after_new_pos", "before_new_source", "after_new_source"))

#colnames(junk)
# drop the original factor variable
junk <- junk[, -which(colnames(junk) %in% c("new_word", "new_source", "before_new_word", "tid", "sentiment", "before_sentiment", "after_sentiment", "before_new_words", "before_new_source", "after_new_word", "after_new_source", "new_relations", "new_pos", "before_new_relations", "before_new_pos", "after_new_relations", "after_new_pos"))]

df <- junk
df <- df[-which(df$CodeType == ""),]
df <- na.omit(df)
df$CodeType <- as.numeric(df$CodeType)
df$CodeType <- df$CodeType -2

sample$CodeType <- junk$CodeType
sample <- sample[-which(sample$CodeType == ""),]
sample <- na.omit(sample)
sample <- sample[rownames(df),]
sample$Code <- df$CodeType


#  prep for writing to file
sample[,1] <- as.character( sample[,1] )
sample$CodeType <- as.character( sample$CodeType )

### sample is actually the core data
### df is the text features (tfidf?)

write.csv(sample, "data/step1_data_sample.csv", row.names=FALSE)
write.csv(df, "data/step1_data_df.csv", row.names=FALSE)
