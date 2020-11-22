#! /bin/env python

### sources incl
### https://towardsdatascience.com/working-with-hugging-face-transformers-and-tf-2-0-89bf35e3555a
### http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
### mainly the latter
### notebook version: https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

# Data
print("Load data")
df = pd.read_csv('data/train.tsv', delimiter='\t', header=None)
df = df.append( pd.DataFrame(data=(("after stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank.", 1),)), ignore_index=True)
df = df.append( pd.DataFrame(data=(("after", 1),)), ignore_index=True)
batch_1 = df[5000:].reset_index()
batch_1[1].value_counts()

# Pretrained model
print("Load model")
# For DistilBERT:
print( "  Model: distilbert-base-uncased")
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
## Want BERT instead of distilBERT? Uncomment the following line:
#print( "  Model: bert-base-uncased")
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
#print( "  Model: roberta-base")
#model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.RobertaTokenizer, 'roberta-base')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

print("Prep data")
# Tokenization
tokenized_raw = batch_1[0].apply((lambda x: tokenizer.tokenize(x)))
tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# Padding
max_len = 0
for i in tokenized.values:
  if len(i) > max_len:
    max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

print("  check padding: ", np.array(padded).shape)

# Masking
attention_mask = np.where(padded != 0, 1, 0)
print("  check mask:", attention_mask.shape)

print("Get embeddings (this is the slow part seems)")
# Querying the model
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
  last_hidden_states, hidden_states = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

#features = last_hidden_states[0][:,0,:].numpy()
features = np.concatenate((hidden_states[6].numpy(),
                            hidden_states[5].numpy(),
                            hidden_states[4].numpy(),
                            hidden_states[3].numpy()),
                            axis=2)
labels = batch_1[1]

#Classifier model 
print("Fit classifier")
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

# Optimizing regularization
# parameters = {'C': np.linspace(0.0001, 100, 20)}
# grid_search = GridSearchCV(LogisticRegression(), parameters)
# grid_search.fit(train_features, train_labels)
#
# print('best parameters: ', grid_search.best_params_)
# print('best scrores: ', grid_search.best_score_)

# Fitting
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

# Evaluating
lr_clf.score(test_features, test_labels)
scores = cross_val_score(lr_clf, train_features, train_labels)
print("  Classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Null model
from sklearn.dummy import DummyClassifier
clf = DummyClassifier()

scores_null = cross_val_score(clf, test_features, test_labels)
print("  Null classifier score: %0.3f (+/- %0.2f)" % (scores_null.mean(), scores_null.std() * 2))

### Distance features of the bank sentence, code from
#https://medium.com/@dhartidhami/understanding-bert-word-embeddings-7dc4d2ea54ca
# >>> 1-cosine(features[1920,6,768:1536], features[1920,10,768:1536])
# 0.9564714431762695
# >>> 1-cosine(features[1920,6,768:1536], features[1920,19,768:1536])
# 0.8140357732772827
# >>> 1-cosine(features[1920,10,768:1536], features[1920,19,768:1536])
# 0.792639970779419
