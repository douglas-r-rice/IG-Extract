# igextract
This repository contains the following:\
(1) data directory: this contains fpc_files (coded Food Policy Council documents in a format appropriate for our analysis) and intermediate files.
(2) three files of the pipeline: s01, s02, and s03, written in r and python. 
(3) small self contained bert demo with links to resources.
(4) supporting notes, readme, makefile, etc

## Running
Run pipeline like so:
```
	echo step 1: prep IG text for analysis. former first half of doug script
	Rscript s01_text_prep.r
	echo step 2: prepare data for bert embeddings and get embeddings. should be two scripts
	python -v s02_embedding_features.py
	echo step 3: fit model to IG text
	Rscript s03_dr_DeepLearner.r
```

## Installing
But first, install like so:
```
	echo first install anaconda.
	echo then import environment 
	conda env create -f environment.yml
	conda activate ig
	echo then keep installing r packages until text_prep.r runs
	echo then keep pip installing python packages until embedding_features.py runsj
	echo (prob none: prob runs fine)
	echo then keep installing r packages until deeplearner.r runs (prob none)
	echo then run the scripts in order
```

if that doesn't work, let file an issue and also try these snippets of  manual instructions
### Installing Keras for R
To run the code, you'll need to setup [Keras](https://keras.rstudio.com/) in RStudio. 

To install the Keras R package from GitHub and get analysis up, execute the following commands:
```
# install necessary packages
install.packages( c("devtools", "cleanNLP", "rJava", "plyr", "dplyr", "magrittr", "readxl", "stringr", "readr", "glue", "e1071", "forcats", "tidyverse", "fastDummies", "caret", "purrr", "xgboost"))
devtools::install_github("rstudio/keras")

library(keras)

# install Keras with TensorFlow backend
install_keras()
```

If it works you should see output ending like this, the accuracy of a neural network set to predict the IG labels of each word:
```
Accuracy with neural network  89.953
Accuracy with neural network (w/out stop words) 90.086
 F1: Aim Attribute Condition Deontic Object Orelse
 F1: 0.84127 0.92647 0.90533 0.86486 0.90435 0.8
```

## Performance
The BERT step drives all system requirements: space (10s of gigs), RAM (with some tricks and hacks I was able to get this to run on a 32GB RAM laptop without crashing python (distilbert-base-uncased, which might be as good as it gets without a proper ML machine). Step 2 is the slowest and takes about 30 mins with 6 cores. I don't know if default install is leveraging GPU or if this is all CPU by default, but I think the latter.


## TODO
* separate step2 into two scripts, one for prep and one for the fitting.
* figure out use of GPU
* include scripts that produce current fpc files
* try switching current step 1 features for tfidf features. then upgrade that tfidf to bi-grams

