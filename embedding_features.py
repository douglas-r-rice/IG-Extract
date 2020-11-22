#! /bin/env python

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
df2 = pd.read_csv('data/train.tsv', delimiter='\t', header=None)

df = pd.read_csv( "data/step1_data_sample.csv")
df = df.rename(columns=dict(zip(df.columns,[0,1,2])))
#print(df.shape)
#batch_1 = df[:2000].reset_index()
batch_1 = df
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

if True: # get last layer
  features = last_hidden_states[:,1,:].numpy()
if False: # get second to last layer
  features = hidden_states[-2][:,1,:].numpy()
if False: # get last four layers
  features = np.concatenate((hidden_states[6][:,1,:].numpy(),
                            hidden_states[5][:,1,:].numpy(),
                            hidden_states[4][:,1,:].numpy(),
                            hidden_states[3][:,1,:].numpy()),
                            axis=1)
if False: # get last four layers plus sentence embedding, secodn to last
  features = np.concatenate((hidden_states[5][:,0,:].numpy()),
                            hidden_states[6][:,1,:].numpy(),
                            hidden_states[5][:,1,:].numpy(),
                            hidden_states[4][:,1,:].numpy(),
                            hidden_states[3][:,1,:].numpy(),
                            axis=1)
# sanity check model output format
assert( np.all(hidden_states[-1].numpy() == last_hidden_states[:,:,:].numpy()) )
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

np.savetxt("data/step2_embedding_features.csv", features, delimiter=",", header=",".join(map(str,range(features.shape[1]))))

if False:
  ## questions being asked.
  ## Some of the single tokens from this dataset are being divided into nine fragments?  What's up?  Looks like date or section references:
  tokenized_raw[[i for i,v in enumerate(attention_mask.numpy()) if all(v) ]]

