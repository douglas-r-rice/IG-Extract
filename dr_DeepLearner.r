# dr_DeepLearner.s
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
library(keras)

use_boosting <- TRUE
use_boosting <- FALSE
#use_session_with_seed(2020)

# load data
### df is the text features (tfidf?), but also the main thing that gets trained
### sample is actually the core data, with the text labels, but is only used at the end reporting accuracy etc
sample <- read.csv( "data/step1_data_sample.csv")
embeddings <- read.csv( "data/step2_embedding_features.csv")
df <- read.csv( "data/step1_data_df.csv")
df <- cbind(df, embeddings)
#df <- cbind(CodeType=df[,1], embeddings)

# as part of reading, convert back to factor
sample[,1] <- factor( sample[,1] )
sample$CodeType <- factor( sample$CodeType )

# index <- createDataPartition(df$CodeType, p=0.85, list=FALSE)
# final.training <- df[index,]
# final.test <- df[-index,]

# randomize order to make the following steps "random"
newOrder <- sample(1:nrow(df))
sample <- sample[newOrder,]
df <- df[newOrder,]

final.training <- df[1:floor(0.9*nrow(df)),]
final.test <- df[ceiling(0.9*nrow(df)):nrow(df),]

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

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = ncol(X_train)) %>% 
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
  X_train, y_train, 
  epochs = 10, 
  batch_size = 64,
  validation_split = 0.1,
  verbose = 1
)

score <- model %>% evaluate(
  X_test, y_test,
  batch_size = 64
)


train_probs <- model %>% predict(
  X_train
)
test_probs <- model %>% predict(
  X_test
)

train_probs <- data.frame(train_probs)
test_probs <- data.frame(test_probs)

X_train_new <- cbind(X_train, train_probs)
X_test_new <- cbind(X_test, test_probs)


train_sample <- sample[1:floor(0.9*nrow(df)),]
test_sample <- sample[ceiling(0.9*nrow(df)):nrow(df),]
compare_data <- test_sample

y_pred_neural <- predict_classes(model, X_test)
compare_data$predicted_1st_round <- y_pred_neural
compare_data$predicted_1st_round_label <- mapvalues(compare_data$predicted_1st_round,
                                                    from = c(0,1,2,3,4,5),
                                                    to = c("Aim", "Attribute", "Condition", "Deontic", "Object", "Orelse"))


myClasses <- as.matrix(table(compare_data$CodeType, compare_data$predicted_1st_round_label))
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

# =-=-=-=-=-=-=-
# compute w/out stop words
# =-=-=-=-=-=-=- 

short_data <- compare_data
colnames(short_data)[1] <- "word"
myStops <- c("the", "of", "a", "an", "and", "is", "it")
myDrops <- which(short_data$word %in% myStops)
short_data <- short_data[-myDrops,]

cat("Accuracy with neural network ", sum(compare_data$Code ==compare_data$predicted_1st_round)/nrow(compare_data)*100, "\n")
cat("Accuracy with neural network (w/out stop words)", sum(short_data$Code ==short_data$predicted_1st_round)/nrow(short_data)*100, "\n")
cat(" F1:", colnames(myClasses ), "\n")
cat(" F1:", f1, "\n")

if (use_boosting) {
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

    # compare_data contains results from both the models for words selected in the test set 
    rownames(compare_data) <- 1:nrow(compare_data)


    myClasses <- as.matrix(table(compare_data$CodeType, compare_data$predicted_2nd_round_label))
    #myClasses <- myClasses[-1,] ### refres to the factor label for "", which has already been filtered out
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



    # =-=-=-=-=-=-=-
    # compute w/out stop words
    # =-=-=-=-=-=-=- 

    short_data <- compare_data
    colnames(short_data)[1] <- "word"
    myStops <- c("the", "of", "a", "an", "and", "is", "it")
    myDrops <- which(short_data$word %in% myStops)
    short_data <- short_data[-myDrops,]

    cat("Accuracy with xgboost using class probabilities ", sum(compare_data$Code == compare_data$predicted_2nd_round)/nrow(compare_data)*100, "\n")
    cat("Accuracy with xgboost using class probabilities ( w/out stop words )", sum(short_data$Code == short_data$predicted_2nd_round)/nrow(short_data)*100, "\n")
cat(" F1:", colnames(myClasses ), "\n")
cat(" F1:", f1, "\n")



}
