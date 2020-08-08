# igextract
This repository contains the following:\
(1) fpc_files directory: this contains coded Food Policy Council documents in a format appropriate for our analysis.\
(2) dr_DeepLearner.r: this file contains the R code that creates the data contained in the fpc_files folder, and then
	executes a series of analyses. The code is commented to indicate what is being done.
(3) RoBERTa_Word_Embeddings.py: this file outputs the rbl.csv which contains the word embeddings obtained from using the second-to-last-layer of the roBERTa model.
(4) DeepLearner_with_roBERTa.r: modified version of dr_DeepLearner.r that incorporates roBERTa word embeddings for analysis.

## Instructions
* Clone the project repo from GitLab with ```git clone```. In this case, run 
```
git clone https://gitlab.com/c2lab1/freygroup/igextract.git
```
Now you have a local copy of the repo on your computer. You can use ```git pull``` to download changes from the remote repo to your local repo.
* Run the file RoBERTa_Word_Embeddings.py to obtain the rbl.csv that will be used in DeepLearner_with_roBERTa.r.
* To run the analysis once: 
```
source('DeepLearner_with_roBERTa.r')
```

* Note that the number of runs depends on the number of random seeds generated. The code by default generates one random seed value and runs the analysis once. Beware that running the analysis once could take a while.
* Please check your tensorflow version before using use_session_with_seed(). The function is currently incompatible with Tensorflow 2.0. If you run into an error, try downgrading the tensorflow version. 
