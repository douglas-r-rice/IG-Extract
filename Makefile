commands:
	echo step 1: prep IG text for analysis. former first half of doug script
	Rscript s01_text_prep.r
	echo step 2: prepare data for bert embeddings and get embeddings. should be two scripts
	python -v s02_embedding_features.py
	echo step 3: fit model to IG text
	Rscript s03_dr_DeepLearner.r

install:
	echo first install anaconda.
	echo then import environment 
	conda env create -f environment.yml
	echo then keep installing r packages until text_prep.r runs
	echo then keep pip installing python packages until embedding_features.py runsj
	echo (prob none: prob runs fine)
	echo then keep installing r packages until deeplearner.r runs (prob none)
	echo then run the scripts in order

embedding_tutorial:
	echo run mostly self-contained bert tutorial cribbed from net
	python -v embedding_features_tutorial.py
