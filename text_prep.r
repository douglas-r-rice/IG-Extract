### TODO
### convert binary presence features to tfidf?
### tag bert on as well

library(stringr)
library(glue)
library(dplyr)
library(fastDummies)
library(sqldf)
options(gsubfn.engine = "R") #from https://github.com/ggrothendieck/sqldf/blob/master/INSTALL
library(assertthat)
"%ni%" <- Negate("%in%")

# set working directory to parent of fpc_files
#setwd(".")

num_files_input <- 5
num_files_input <- 19
preserve_old_analysis <- FALSE

# read in some sample data
path = "fpc_files/fpc_{i}_with_code.csv"

fpc1 <- data.frame()
for (i in 1:num_files_input) {
  fpc1 <- rbind(fpc1, read.csv(glue(path)))
}

##3 slow but easy repair of sid (which in prior code is equal to 1 always --- not doing what it should.  can recover because tid is sound)
l_tid = 0
l_sid = 0
for (i in 1:nrow(fpc1)) {
    if (fpc1[i,"tid"] < l_tid) {
        l_sid = l_sid + 1
    }
    fpc1[i,"sid"] <- l_sid
    l_tid <- fpc1[i,"tid"]
}

#filter out longest sentences, hopefully to make embeddings possible
#  the final hidden states output by the model are the dimensionality of the longest sentences, with dramatic increases in size of representation with an increase in setnence length.  removing the longest sentence draamaatically reduce memory load, the difference between this running on a laptop or not.
# from df[tokenized_raw.apply(len) > 200].sid.unique()
sid_toolong = c( 26,  87, 117, 296, 446)
#fpc1 = fpc1[df.sid.apply(lambda x : x not in sid_toolong)].reset_index(drop=True)
fpc1 = subset(fpc1, sid %ni% sid_toolong)

### redo to keep sid's consecutive
##3 slow but easy repair of sid (which in prior code is equal to 1 always --- not doing what it should.  can recover because tid is sound)
l_tid = 0
l_sid = 0
for (i in 1:nrow(fpc1)) {
    if (fpc1[i,"tid"] < l_tid) {
        l_sid = l_sid + 1
    }
    fpc1[i,"sid"] <- l_sid
    l_tid <- fpc1[i,"tid"]
}


### this is a now irrelevant part of aligning spacy tokenization here with bert tokenization later, for merging on those features 
fpc1$tid_base <- fpc1$tid
fpc1$tid_adjust <- 0
if (!preserve_old_analysis) {
    ### also repair tid's to remove them from discarded words
    ### lots of checking here, pretty frustrating
    l_tid <- 0
    tids_skipped <- 0
    l_sid <- -1
    for (i in 1:nrow(fpc1)) {
        #for (i in 1:200) {
        #print( fpc1[i,])
        ### detect that we're now at a new sentence.  If so, do some santiy checks and reset the token id trackers and increment the sid tracker
        if (fpc1[i,"sid"] != l_sid) {
            #print("in 1")
            #print( c(i, l_sid, fpc1[i,"sid"], fpc1[i,"CodeType"], fpc1[i,"word"]))
            #print( fpc1[i,"sid_text"])
            # double check that my sid count is stable
            assert_that(fpc1[i,"sid"] == l_sid + 1)
            #print( c(i, l_tid, fpc1[i,"tid"], tids_skipped))
            # check that the token id has just reset to 0
            assert_that(fpc1[i,"tid"] == 0)
            # also check that either token id has just reset to lower than it was at the end of the last sentence, or, in the case that the whole last sentence was skipped, that l_tid is still zero from it's last reset at the beginning of the last sentence (there are many wholly skipped sentences, confirmed in the if statement below)
            assert_that((fpc1[i,"tid"] < l_tid) | (l_tid == 0))
            if (l_tid == 0){
                #print("in 2")
                #print( "skipped sentence")
                #print(i-1)
                #print(fpc1[i-1,])
            }
            #print("")
            l_sid <- l_sid + 1
            l_tid <- 0
            tids_skipped <- 0
        }
        if (fpc1[i,"CodeType"] == "") {
            #print("in 3")
            fpc1[i,"tid"] <- -1
            fpc1[i,"tid_adjust"] <- -1
            tids_skipped <- tids_skipped + 1
        } else {
            #print("in 4")
            fpc1[i,"tid_adjust"] <- tids_skipped
            fpc1[i,"tid"] <- fpc1[i,"tid"] - tids_skipped + 1
            l_tid <- l_tid + 1
        }
        #print( c(i, l_sid, fpc1[i,"sid"]))
        #print( c(i, l_tid, fpc1[i,"tid"], tids_skipped))
        ### confirm either that my tid tracking is working or that this line was skipped
        assert_that(l_tid == fpc1[i,"tid"] | fpc1[i,"tid"] == -1) 
        ### confirm that the sid's still line up
        assert_that(l_sid == fpc1[i,"sid"])
        #print( fpc1[i,])
        #print('')
    }
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

assert_that( (nrow(fpc1) - nrow(na.omit(fpc1))) == max(fpc1$sid)+1)
if (FALSE | !preserve_old_analysis) {
    ### this gets rid of the ROOT elements, nothing more.  about 560: equal to the number of sentences
    fpc1 <- na.omit(fpc1)
}

assert_that(length(fpc1$index) == length(unique(fpc1$index)))

# create lag variables
junktmp <- fpc1 
junktmp$tid_base <- NULL
junktmp$tid_adjust <- NULL
junktmp$sid <- NULL
junktmp$sid_text <- NULL
assert_that(length(junktmp$index) == length(unique(junktmp$index)))

lagjunk <- junktmp
lagjunk$tid <- junktmp$tid+1
lagjunk$index <- junktmp$index+1
colnames(lagjunk)[2:(ncol(lagjunk)-1)] <- paste0("before_", colnames(lagjunk)[2:(ncol(lagjunk)-1)])
if (FALSE | !preserve_old_analysis) {
    lagjunk <- na.omit(lagjunk)
}
assert_that(length(lagjunk$index) == length(unique(lagjunk$index)))
junk <- merge(junktmp, lagjunk, by=c("index", "tid"), all.x=T)
assert_that(length(junk$index) == length(unique(junk$index)))

# create lead variables
leadjunk <- junktmp
leadjunk$tid <- junktmp$tid-1
leadjunk$index <- junktmp$index-1
colnames(leadjunk)[2:(ncol(leadjunk)-1)] <- paste0("after_", colnames(leadjunk)[2:(ncol(leadjunk)-1)])
if (FALSE | !preserve_old_analysis) {
    leadjunk <- na.omit(leadjunk)
}
assert_that(length(leadjunk$index) == length(unique(leadjunk$index)))
assert_that(length(junk$index) == length(unique(junk$index)))
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
if (preserve_old_analysis) {
    ### EXPERIMENT WITH NOT OMITTING NAS
    df <- na.omit(df)
}
df$CodeType <- as.numeric(df$CodeType)
df$CodeType <- df$CodeType -2

sample <- data.frame(fpc1[, c("new_word", "sid", "sid_text", "tid", "tid_base", "tid_adjust" )])
sample$CodeType <- junk$CodeType
sample <- sample[-which(sample$CodeType == ""),]
if (preserve_old_analysis) {
    sample <- na.omit(sample)
}
assert_that(-1 %ni% sample$tid )
assert_that(all(dim(sample) == dim(na.omit(sample))))
assert_that(all(sample$CodeType == na.omit(sample)$CodeType))
if (preserve_old_analysis) {
    sample <- sample[rownames(df),]
}
sample$Code <- df$CodeType

### sanity check
assert_that(nrow(df) == nrow(sample))
if (preserve_old_analysis) {
    assert_that(
        ### for the other (pre experiment of removing NA's)
        all(df$CodeType == sample$Code) 
    )# confirm rows actually aligned right after all this
} else {
    assert_that(
        ### for one data format (post experiment of removing NA's)
        all(as.numeric(df$CodeType) == as.numeric(sample$CodeType)-2)
    ) # confirm rows actually aligned right after all this
}
all(as.numeric(df$CodeType) == as.numeric(sample$CodeType)-2)
all(df$CodeType == sample$Code)

#  prep for writing to file
sample[,1] <- as.character( sample[,1] )
sample$CodeType <- as.character( sample$CodeType )
sample$Code <- as.character( sample$Code )

### sample is actually the core data
### df is the text features (tfidf?)

print("final sizes")
print(c("sample:", dim(sample)))
print(c("df:", dim(df)))

write.csv(sample, "data/step1_data_sample.csv", row.names=FALSE)
write.csv(df, "data/step1_data_df.csv", row.names=FALSE)
