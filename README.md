# igextract
This repository contains the following:\
(1) fpc_files directory: this contains coded Food Policy Council documents in a format appropriate for our analysis.\
(2) dr_DeepLearner.r: this file contains the R code that creates the data contained in the fpc_files folder, and then
	executes a series of analyses. The code is commented to indicate what is being done.


## Installing Keras for R
To run the code, you'll need to setup [Keras](https://keras.rstudio.com/) in RStudio. 

To install the Keras R package from GitHub and get analysis up, execute the following commands:
```
# necessary packages
install.packages( c("devtools", "cleanNLP", "rJava", "plyr", "dplyr", "magrittr", "readxl", "stringr", "readr", "glue", "e1071", "forcats", "tidyverse", "fastDummies", "caret", "purrr", "xgboost"))
devtools::install_github("rstudio/keras")

library(keras)

#install Keras with TensorFlow backend
install_keras()
```

The you can run the analysis
```
source('dr_DeepLearner.r')
```

If it works you should see output ending like this, the accuracy of a neural network set to predict the IG labels of each word:
```
Accuracy with neural network  71.909
Accuracy with xgboost using class probabilities  73.102
Accuracy with neural network  72.936
Accuracy with xgboost using class probabilities  74.019
```
