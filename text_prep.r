### TODO
### convert binary presence features to tfidf?
### tag bert on as well

library(stringr)
library(glue)
library(dplyr)
library(fastDummies)
library(sqldf)

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

##3 slow but easy repair of sid (which is equal to 1 always, not doing what it should.  can recover because tid is sound)
l_tid = 0
l_sid = 0
for (i in 1:nrow(fpc1)) {
    if (fpc1[i,"tid"] < l_tid) {
        l_sid = l_sid + 1
    }
    fpc1[i,"sid"] <- l_sid
    l_tid <- fpc1[i,"tid"]
}

# pull just the data we want
fpc1$new_relations <- as.factor(fpc1$relation)
fpc1$new_word <- as.factor(fpc1$word)
fpc1$new_pos <- as.factor(fpc1$pos)
fpc1$new_source <- as.factor(fpc1$word_source)

sentences <- sqldf("SELECT sid, GROUP_CONCAT(word, ' ') AS sid_text FROM fpc1 GROUP BY sid;")
fpc1 <- merge(fpc1, sentences, by="sid")


fpc1$CodeType <- as.factor(fpc1$CodeType)

fpc1$id <- NULL
#fpc1$sid <- NULL
fpc1$index <- 0
for (i in 1:nrow(fpc1)) {
  fpc1[i, "index"] <- i-1
}
sample <- data.frame(fpc1[, c("new_word", "sid", "sid_text")])

# create lag variables
lagjunk <- fpc1 
lagjunk$tid <- fpc1$tid+1
lagjunk$index <- fpc1$index+1
colnames(lagjunk)[3:(ncol(lagjunk)-2)] <- paste0("before_", colnames(lagjunk)[3:(ncol(lagjunk)-2)])
junk <- merge(fpc1, lagjunk, by=c("index", "tid", "sid", "sid_text"), all.x=T)

# create lead variables
leadjunk <- fpc1 
leadjunk$tid <- fpc1$tid-1
leadjunk$index <- fpc1$index-1
colnames(leadjunk)[3:(ncol(lagjunk)-2)] <- paste0("after_", colnames(leadjunk)[3:(ncol(lagjunk)-2)])
junk <- merge(junk, leadjunk, by=c("index", "tid", "sid", "sid_text"), all.x=T)

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
