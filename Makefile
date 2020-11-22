commands:
	echo run bert tutorial cribbed from net
	python -v embedding_features_tutorial.py
	echo prep IG text for analysis
	Rscript text_prep.r
	echo fit model to IG text
	Rscript dr_DeepLearner.r
