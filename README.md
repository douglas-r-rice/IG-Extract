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
Important: Please make sure that you have installed a version of Java (i.e. 64-bit Java or 32-bit Java) that is the same as your R version (i.e. 64-bit R or 32-bit R) prior to installing the rJava package. Otherwise, you may encounter the error described [here](https://www.r-statistics.com/2012/08/how-to-load-the-rjava-package-after-the-error-java_home-cannot-be-determined-from-the-registry/). To check the type of R version you are using, type into the RStudio terminal: 
```
Sys.info()[["machine"]]  
```

### Download Anaconda
Anaconda is a Python distribution designed for predictive analytics and scientific computing. Anaconda includes many data science and machine learning packages. The best way to set up Python 3 and Jupyter Notebook is to get [Anaconda](https://www.anaconda.com/download/). It is recommended to stay one version back from the latest Python to avoid incompatibilty issues with other packages. 

### Hugging Face Transformers Installation
The first step to installing the Hugging Face library is to ensure that you have Tensorflow 2.0 and/or PyTorch installed in your [conda environment](https://medium.com/@margaretmz/anaconda-jupyter-notebook-tensorflow-and-keras-b91f381405f8). You can install [Pytorch](https://pytorch.org/get-started/locally/#mac-anaconda) via Anaconda or pip. Run the command presented to you in the Anaconda prompt after selecting the appropriate configuration options. 

You can install Transformers in two ways: (1) [with pip](https://huggingface.co/transformers/installation.html#installation-with-pip) or (2) [from source](https://huggingface.co/transformers/installation.html#installing-from-source). The [Transformers documentation](https://huggingface.co/transformers/) is a good resource for getting started and a good reference for Transformers' built-in features.

```
# create a conda virtual environment
conda create -n [env_name] python = [python_version]

# activate the environment
conda activate [env_name]

# install PyTorch inside the environment
pip install torchvision

# install huggingface library
pip install transformers

```

### Install Git
[Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) is a distributed version control system. You will be using git to get the files in this repository. You can run git commands in the shell. 

## Instructions
* Clone the project repo from GitLab with ```git clone```. In this case, run 
```
git clone https://gitlab.com/c2lab1/freygroup/igextract.git
```
Now you have a local copy of the repo on your computer. You can use ```git pull``` to download changes from the remote repo to your local repo.
* Run the file BERT_Embeddings.py to obtain the bert_emb.csv that will be used in DeepLearner_with_BERT.r.
* Then run the analysis ```source('DeepLearner_with_BERT.r')```.

Your output should be something along the lines of: 
```
Accuracy with neural network  70.60738
Accuracy with xgboost using class probabilities  72.66811
Accuracy with neural network  71.68263
Accuracy with xgboost using class probabilities  74.14501
```