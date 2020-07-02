# igextract
This repository contains the following:\
(1) fpc_files directory: this contains coded Food Policy Council documents in a format appropriate for our analysis.\
(2) BERT_Embeddings.py: this contains the code to obtain the word embeddings from BERT for all the data in the fpc_files directory. \
(3) DeepLearner_with_BERT.r: modified version of dr_DeepLearner.r that incorporates the embeddings for analyses. \
(4) dr_DeepLearner.r: this file contains the R code that creates the data contained in the fpc_files folder, and then
	executes a series of analyses. The code is commented to indicate what is being done. 


## Software Installation for Deep Learning
## Installing Keras for R
To run the code, you'll need to setup [Keras](https://keras.rstudio.com/) in RStudio. 

To install the Keras R package from GitHub and get analysis up, execute the following commands:
```
# install necessary packages
install.packages(c("devtools", "cleanNLP", "rJava", "plyr", "dplyr", "magrittr", "readxl", "stringr", "readr", "glue", "e1071", "forcats", "tidyverse", "fastDummies", "caret", "purrr", "xgboost"))
devtools::install_github("rstudio/keras")

library(keras)

# install Keras with TensorFlow backend 
install_keras()
```

### Download Anaconda
Anaconda is a Python distribution designed for predictive analytics and scientific computing. Anaconda includes many data science and machine learning packages. The best way to set up Python 3 and Jupyter Notebook is to get [Anaconda](https://www.anaconda.com/download/). It is recommended to stay one version back from the latest Python to avoid incompatibilty issues with other packages. 

Please refer to this [article](https://medium.com/@margaretmz/anaconda-jupyter-notebook-tensorflow-and-keras-b91f381405f8) for more details on how to install Keras and TensorFlow in Python and on how to create a virtual environment. 

### Hugging Face Transformers Installation
The first step to installing the Hugging Face library is to ensure that you have Tensorflow 2.0 and/or PyTorch installed. You can install Pytorch via Anaconda. 
Visit [this](https://pytorch.org/get-started/locally/#mac-anaconda) website. Run the command presented to you in the Anaconda prompt after selecting the appropriate configuration options. 

You can install Transformers in two ways: (1) [with pip](https://huggingface.co/transformers/installation.html#installation-with-pip) or (2) [from source](https://huggingface.co/transformers/installation.html#installing-from-source). The [Transformers documentation](https://huggingface.co/transformers/) is a good resource for getting started and a good reference for Transformers' built-in features.

### Install Git
[Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) is a distributed version control system. You will be using git to get the files in this repository. You can run git commands in the shell. 

## Instructions
* Clone the project repo from GitLab with ```git clone```. In this case, run 
```
git clone https://gitlab.com/c2lab1/freygroup/igextract.git
```
Now you have a local copy of the repo on your computer. You can use ```git pull``` to download changes from the remote repo to your local repo.
* Run the file BERT_Embeddings.py to obtain the feature_embeddings.csv that will be used in DeepLearner_with_BERT.r.
* Then run the analysis ```source('DeepLearner_with_BERT.r')```.

Your output ending should be something along the lines of: 
```
Accuracy with neural network  72.56
Accuracy with xgboost using class probabilities  73.319
Accuracy with neural network  73.75
Accuracy with xgboost using class probabilities  73.889
```