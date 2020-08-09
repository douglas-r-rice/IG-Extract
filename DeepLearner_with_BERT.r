# DeepLearner_with_BERT.s
# This program using a deep learning model to predict classifications.
#
# dr 1.29.2020

# front end matters
library(cleanNLP)
library(rJava)
library(plyr)
library(dplyr)
library(magrittr)
library(readxl)
library(stringr)
library(readr)
library(glue)
library(e1071)
library(forcats)
library(tidyverse)
library(fastDummies)
library(caret)
library(purrr)
library(xgboost)
library(tensorflow)
library(keras)

# optional: set seed for reproducibility
# use_session_with_seed(2020)

# load BERT embeddings
emb_path = "bert_emb.csv"
emb_df <- read.csv(emb_path)

# read in some sample data
path = "fpc_files/fpc_{i}_with_code.csv"

fpc1 <- data.frame()
for (i in 1:19) {
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
sample <- data.frame(junk[, "new_word"])
junk$new_word <- as.numeric(junk$new_word)
junk$new_source <- as.numeric(junk$new_source)
junk$before_new_word <- as.numeric(junk$before_new_word)
junk$before_new_source <- as.numeric(junk$before_new_source)
junk$after_new_word <- as.numeric(junk$after_new_word)
junk$after_new_source <- as.numeric(junk$after_new_source)

junk <- dummy_cols(junk, select_columns = c("new_word", "before_new_word", "after_new_word", "new_relations", "new_pos", "before_new_relations", "before_new_pos", "after_new_relations", "after_new_pos", "before_new_source", "after_new_source"))

# colnames(junk)
# drop the original factor variable
junk <- junk[, -which(colnames(junk) %in% c("new_word", "new_source", "before_new_word", "tid", "sentiment", "before_sentiment", "after_sentiment", "before_new_words", "before_new_source", "after_new_word", "after_new_source", "new_relations", "new_pos", "before_new_relations", "before_new_pos", "after_new_relations", "after_new_pos"))]

sample$CodeType <- junk$CodeType
if (TRUE) {
    df <- cbind(junk , emb_df)
} 
if (FALSE) {
    df <- junk
}
df <- df[-which(df$CodeType == ""),] 
sample <- sample[-which(sample$CodeType == ""),]
sample <- na.omit(sample)
df <- na.omit(df) 

sample <- sample[rownames(df),]

df$CodeType <- as.numeric(df$CodeType)
df$CodeType <- df$CodeType -2
sample$Code <- df$CodeType

# index <- createDataPartition(df$CodeType, p=0.85, list=FALSE)
# final.training <- df[index,]
# final.test <- df[-index,]

# randomize order 
newOrder <- sample(1:nrow(df))
sample <- sample[newOrder,]
df <- df[newOrder,]

final.training <- df[1:8320,] 
final.test <- df[8321:nrow(df),] 

X_train <- final.training %>% 
  select(-CodeType) %>%
  scale() 

X_train <- t(na.omit(t(X_train))) 

y_train <- final.training$CodeType

X_test <- final.test %>% 
  select(-CodeType) %>%
  scale() 

X_test <- t(na.omit(t(X_test))) 

X_train <- X_train[ ,intersect(colnames(X_train), colnames(X_test))] 
X_test <- X_test[ ,intersect(colnames(X_train), colnames(X_test))] 

y_test <- final.test$CodeType



if (FALSE) {
    # repeat the procedure above for emb_df
    emb_df$CodeType <- junk$CodeType
    emb_df <- emb_df[-which(emb_df$CodeType == ""),] # remove rows with no codetype
    emb_df <- na.omit(emb_df) # removes nan 
    emb_df <- emb_df[rownames(df),] # subset rows only in df 
    emb_final.training <- emb_df[1:8320,] 
    emb_final.test <- emb_df[8321:nrow(emb_df),] 

    # scaling
    emb_X_train <- emb_final.training %>% 
    select(-CodeType) %>%
    scale()

    emb_X_test <- emb_final.test %>% 
    select(-CodeType) %>%
    scale()

    final_X_train <- cbind(X_train, emb_X_train)
    final_X_test <- cbind(X_test, emb_X_test)
}
if (TRUE) {
    final_X_train <- X_train
    final_X_test <- X_test
}



model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = ncol(final_X_train)) %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 6, activation = 'softmax')

model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

model %>% fit(
  final_X_train, y_train, 
  epochs = 10, 
  batch_size = 64,
  validation_split = 0.1,
  verbose = 1
)

score <- model %>% evaluate(
  final_X_test, y_test,
  batch_size = 64
)


train_probs <- model %>% predict(
  final_X_train
)
test_probs <- model %>% predict(
  final_X_test
)

train_probs <- data.frame(train_probs)
test_probs <- data.frame(test_probs)

X_train_new <- cbind(final_X_train, train_probs)
X_test_new <- cbind(final_X_test, test_probs)

train_sample <- sample[1:8320,]
test_sample <- sample[8321:nrow(df),]
compare_data <- test_sample

y_pred_neural <- predict_classes(model, final_X_test)
compare_data$predicted_1st_round <- y_pred_neural
compare_data$predicted_1st_round_label <- mapvalues(compare_data$predicted_1st_round,
                                                    from = c(0,1,2,3,4,5),
                                                    to = c("Aim", "Attribute", "Condition", "Deontic", "Object", "Orelse"))


# compare_data contains results from both the models for words selected in the test set 
rownames(compare_data) <- 1:nrow(compare_data)

cat("Accuracy with neural network ", sum(compare_data$Code ==compare_data$predicted_1st_round)/nrow(compare_data)*100, "\n")



# =-=-=-=-=-=-=-
# compute w/out stop words
# =-=-=-=-=-=-=- 

short_data <- compare_data
colnames(short_data)[1] <- "word"
myStops <- c("the", "of", "a", "an", "and", "is", "it")
myDrops <- which(short_data$word %in% myStops)
short_data <- short_data[-myDrops,]

cat("Accuracy with neural network ", sum(short_data$Code ==short_data$predicted_1st_round)/nrow(short_data)*100, "\n")

# =-=-=-=-=-=-=-
# incl boosting
# =-=-=-=-=-=-=- 
if (FALSE) {
    # training xgboost classifier by adding class probabilities from neural network to the existing data
    xgb <- xgboost(data = data.matrix(X_train_new), 
                   label = y_train, 
                   eta = 0.1,
                   max_depth = 20, 
                   nround=30,
                   lambda=0.08,
                   eval_metric = "mlogloss",
                   objective = "multi:softmax",
                   num_class = 6,
                   nthread = 3
    )

    y_pred_xgb <- predict(xgb, data.matrix(X_test_new))
    compare_data$predicted_2nd_round <- y_pred_xgb
    compare_data$predicted_2nd_round_label <- mapvalues(compare_data$predicted_2nd_round,
                                                        from = c(0,1,2,3,4,5),
                                                        to = c("Aim", "Attribute", "Condition", "Deontic", "Object", "Orelse"))


    myClasses <- as.matrix(table(compare_data$CodeType, compare_data$predicted_1st_round_label))
    myClasses <- myClasses[-1,]
    n <- sum(myClasses)
    nc <- nrow(myClasses)
    myDiag <- diag(myClasses)
    myRowSums <- apply(myClasses, 1, sum)
    myColSums <- apply(myClasses, 2, sum)
    p <-  myRowSums / n
    q <- myColSums / n

    myAccuracy <- sum(myDiag) / n
    precision <- myDiag / myColSums
    recall <- myDiag / myRowSums
    f1 <- (2 * precision * recall ) / (precision + recall)

    myClasses <- as.matrix(table(compare_data$CodeType, compare_data$predicted_2nd_round_label))
    myClasses <- myClasses[-1,]
    n <- sum(myClasses)
    nc <- nrow(myClasses)
    myDiag <- diag(myClasses)
    myRowSums <- apply(myClasses, 1, sum)
    myColSums <- apply(myClasses, 2, sum)
    p <-  myRowSums / n
    q <- myColSums / n

    myAccuracy <- sum(myDiag) / n
    precision <- myDiag / myColSums
    recall <- myDiag / myRowSums
    f12 <- (2 * precision * recall ) / (precision + recall)

    short_data <- compare_data
    colnames(short_data)[1] <- "word"
    myStops <- c("the", "of", "a", "an", "and", "is", "it")
    myDrops <- which(short_data$word %in% myStops)
    short_data <- short_data[-myDrops,]

    nn_one = sum(compare_data$Code ==compare_data$predicted_1st_round)/nrow(compare_data)*100
    xgb_one = sum(compare_data$Code == compare_data$predicted_2nd_round)/nrow(compare_data)*100
    nn_two = sum(short_data$Code ==short_data$predicted_1st_round)/nrow(short_data)*100
    xgb_two = sum(short_data$Code == short_data$predicted_2nd_round)/nrow(short_data)*100
    
    cat("Accuracy with neural network ", nn_one, "\n")
    cat("Accuracy with neural network ", nn_two, "\n")
    cat("Accuracy with xgboost using class probabilities ", xgb_one, "\n")
    cat("Accuracy with xgboost using class probabilities ", xgb_two, "\n")

    new_list = list(nn_one, xgb_one, f1, nn_two, xgb_two, f12)

    print(new_list)
}