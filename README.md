# igextract
This repository contains the following:\
(1) fpc_files directory: this contains coded Food Policy Council documents in a format appropriate for our analysis.\
(2) fpc_all.csv: all the files in the fpc_files directory concatenated into one single file. \
(3) BERT_Embeddings.ipynb: this contains the code to obtain the word embeddings from BERT for all the data in the fpc_files folder. \
(4) DeepLearner_with_BERT.r: modified version of dr_DeepLearner.r that incorporates the embeddings for analyses.


## Software Installation for Deep Learning
### Download Anaconda
Anaconda is a Python distribution designed for predictive analytics and scientific computing. Anaconda includes many data science and machine learning packages. The best way to set up Python 3 and Jupyter Notebook is to get [Anaconda](https://www.anaconda.com/download/). It is recommended to stay one version back from the latest Python to avoid incompatibilty issues with other packages. 

Please refer to this [article](https://medium.com/@margaretmz/anaconda-jupyter-notebook-tensorflow-and-keras-b91f381405f8) for more details on how to install Keras and TensorFlow in Python and on how to create a virtual environment. 

### Hugging Face Transformers Installation
The first step to installing the Hugging Face library is to ensure that you have Tensorflow 2.0 and/or PyTorch installed. You can install Pytorch via Anaconda. 
Visit [this](https://pytorch.org/get-started/locally/#mac-anaconda) website. Run the command presented to you in the Anaconda prompt after selecting the appropriate configuration options. 

You can install Transformers in two ways: (1) [with pip](https://huggingface.co/transformers/installation.html#installation-with-pip) or (2) [from source](https://huggingface.co/transformers/installation.html#installing-from-source). The [Transformers documentation](https://huggingface.co/transformers/) is a good resource for getting started and a good reference for Transformers' built-in features.

## Other Instructions
* Before running DeepLearner_with_BERT.r, you must run the file BERT_Embeddings.ipynb to obtain the feature_embeddings.csv that will be used in DeepLearner_with_BERT.r.
* Install any other necessary packages.
* Change the file path if applicable.
* Run the notebook/R code and get your result displayed.