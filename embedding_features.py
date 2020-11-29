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
df = pd.read_csv( "data/step1_data_sample.csv")
df = df.rename(columns=dict(zip(df.columns,[0, "sid", "sid_text","tid","tid_base","tid_adjust",1,2])))
#print(df.shape)
#batch_1 = df[:2000].reset_index()
### clean up sentences to preempt tokenization problems
### this is all effort to make the space tokenization look more like the bert tokenization, especially special handling of ellipses and hypthens
df = df.assign(sid_text=df['sid_text'].apply(
    lambda x:x.lower().replace('-lrb-', '[').replace('-rrb-', ']').replace('...', '<ELLIPSES>').replace('.', '').replace( '<ELLIPSES>', '...').replace(' - ', ' , ').replace('-', '').replace('  ', ' ').strip()
    ))
#df = df.assign(sid_text_old=df['sid_text'])
#df = df.assign(sid_text=df['sid_text'].apply(
    #lambda x:x.lower().replace('.', '').replace('-', '').replace('  ', ' ').replace('root ', '').strip()
    #))

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
tokenized_raw = batch_1['sid_text'].apply((lambda x: tokenizer.tokenize(x)))
tokenized = batch_1['sid_text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

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

### align tokenizers
### a huge amount of effort wen tin to fixig the problem that the dependency parsing was done with spacy's tokenizer, and the embeddings were with bert's tokenizer.  reconciling them means changes on the bert side (the ## tags), but mostly changes ont eh spacy side or changes on the source text (punctuation ahndling).  exmaples include ellipses, hyphens and apostrophes.
def tokBertRepair(tok):
  '''
  turns first token of a tokenized word into the tokenized word (still keeping ## fragmeents as well)
  '''
  #print( list( tok ))
  for i in reversed(range(len(tok))):
    #print(i)
    #print(i, tok[i])
    #print(i, tok[i], tok[i].startswith('##'))
    if tok[i].startswith('##'):
      tok[i-1] = tok[i-1] + tok[i][2:]
  return( tok )

def tokAligner (tok1):
  """
  produce a list of 0's, 1's, and -1's that can be used to say which index of tok1 maps to the value of tok2.  lots of small tweaks to reconcile the spacy tokenizer with the needs of the bert tokenizer, in the form of a mapping from the former to the relevant index of the latter.
  returns fixed version of list that resenbles the bert version, and also a map to get to that version from the source
  """
  tokMap = [0]* len(tok1)
  for i in reversed(range(len(tok1))):
    if tok1[i] == '...':
      # this goes from being one to three tokens, hence shift by 2
      tokMap[i] += 2
      tok1 = tok1[:i] + ['.']*3 + tok1[(i+1):]
    elif tok1[i].endswith('.'):
      tok1 = tok1[:i] + [tok1[i][:-1], '.'] + tok1[(i+1):]
      tokMap[i] += 1
    if "'s" == tok1[i]:
      tokMap[i] += 1
      tok1 = tok1[:i] + ["'", 's'] + tok1[(i+1):]
    elif "''" == tok1[i]:
      tokMap[i] += 1
      tok1 = tok1[:i] + ["'", "'"] + tok1[(i+1):]
    elif '``' == tok1[i]:
      tokMap[i] += 1
      tok1 = tok1[:i] + ['`', '`'] + tok1[(i+1):]
    elif ':]' == tok1[i]:
      tokMap[i] += 1
      tok1 = tok1[:i] + [':', ']'] + tok1[(i+1):]
    if '/' in tok1[i]:
      tok1 = tok1[:i] + list(tok1[i].partition('/')) + tok1[(i+1):]
      tokMap[i] += 2
  return( tok1, tokMap )

# Implementing/Checking sentence merge
df = df.assign(bertid=-1)  # this is the value for mapping from a pre-labeled token to its embedding
#for s in range(4000):
for s in range(df.shape[0]):
  dep_feats = df.loc[s].values.tolist() # labeled token and its metadata
  sid = dep_feats[1]
  tokid = dep_feats[4]
  #print(dep_feats)
  tokspacy = dep_feats[2].split(' ') # "source" sentence from preprocessing
  tokbert = tokBertRepair(tokenized_raw[s].copy())
  tokbert = [t for t in tokbert if not t.startswith('##')] # version of sentence that will produce embedding
  tokspacy_orig = tokspacy.copy()
  tokspacy, tokmap = tokAligner(tokspacy) # produce alignemnt wbetween two tokenizations
  if False:
    for i in range(len(tokspacy_orig)):
        # print(s, i, tokbert[i])
        # print(s, i, tokbert[i], tokspacy[i], sum(tokmap[:i]))
        # print(s, i, tokbert[i], tokspacy[(i + sum(tokmap[:i]))], sum(tokmap[:i]))
        #assert( tokspacy[i] == tokbert[(i + sum(tokmap[:i]))] )
        if tokspacy_orig[i] != tokbert[(i + sum(tokmap[:i]))]:
            #print(s, i, dep_feats[2])
            #print(s, i, tokspacy_orig)
            #print(s, i, tokspacy)
            #print(s, i, tokbert)
            #print(s, i, tokmap)
            print(s, i, tokspacy_orig[i], tokbert[(i + sum(tokmap[:i]))], sum(tokmap[:i]), (i + sum(tokmap[:i])))
        # print(s, tokspacy)
        # print(s, tokbert)
        #break
  assert( tokbert == tokspacy)
  assert( len(tokspacy_orig) + sum(tokmap) == len(tokbert) )
  ### if alignment worked, very last characters of very last tokens of the original should map ontot he bert
  assert( tokspacy_orig[-1][-1] == tokbert[(len(tokspacy_orig) - 1 + sum(tokmap))][-1])
  if False or not all([v == 0 for v in tokmap]):
    if tokspacy_orig[-1][-1] != tokbert[(len(tokspacy_orig) - 1 + sum(tokmap))][-1]:
      print(s, "a save ")
      print(s, tokspacy_orig)
      print(s, tokspacy)
      print(s, tokbert)
      print(s, tokmap)
      print(s, tokspacy_orig[-1], tokbert[(len(tokspacy_orig) - 1 + sum(tokmap))], sum(tokmap))
      break
  ### continue code from before
  tok_sent = tokbert
  # tweak target token to make it match easier
  word = dep_feats[0].replace('-','').replace('.', '').replace('LRB', '[').replace('RRB', ']').lower().partition('/')[0]
  # adjusted bert index of target token, using map's accumulated location adjustments
  match_word_idx = tokid + sum(tokmap[:(tokid)])
  #print( "x{}x{}x{}x".format(dep_feats[0], word, tok_sent[match_word_idx]))
  # this succeeds in handling all but two of thousands of cases.
  if word == tok_sent[match_word_idx]:
    df.at[s,'bertid'] = match_word_idx
    continue
  else: # the two failures are tokens that made ti through as effectively empty strings (single hyphens). treat them as their preceding word .
    if not word:
      #print("GOT IN", sid, tokid, dep_feats[0])
      df.at[s,'bertid']=df.loc[s-1].values.tolist()[4] + sum(tokmap[:(tokid)])
      continue
    print(sid, word, tok_sent[match_word_idx], tokid, match_word_idx)
    print(sid, tokbert)
    print(sid, tokmap, tokmap[match_word_idx], sum(tokmap[:match_word_idx]))
    print(sid, tokspacy_orig)
    print(sid, dep_feats)

print("Get embeddings (this is the slow part seems)")
# Querying the model
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
  last_hidden_states, hidden_states = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

nwords = hidden_states[0].shape[0]
nsentidxs = hidden_states[0].shape[1]
nfeats = hidden_states[0].shape[2]

### save costly output
torch.save(hidden_states, "data/step25_hidden_states.pt")
#torch.load("data/step25_hidden_states.pt")

if False: # get last layer
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
if False: # get last layer, by word
  mask = torch.tensor([[[v,]*nfeats,] for v in df.bertid])
  features = torch.gather(hidden_states[6][:,:,:], 1, mask)
  ### get to right dimensionality and type
  features  = features[:,0,:].numpy()
  ### tested picking out with this line
  ###  torch.gather(hidden_states[6][::50,:,0], 1, torch.tensor([[v%169,] for v in range(172)]))[-10:]
if False: # get last layer, by word
  mask = torch.tensor([[[v,]*nfeats,] for v in df.bertid])
  features = torch.gather(hidden_states[5][:,:,:], 1, mask)
  features  = features[:,0,:].numpy()
  ### tested picking out with this line
  ###  torch.gather(hidden_states[6][::50,:,0], 1, torch.tensor([[v%169,] for v in range(172)]))[-10:]
if True: # get word embeddings for last four layers plus whole sentence embedding, last layer
  mask = torch.tensor([[[v,]*nfeats,] for v in df.bertid])
  features = np.concatenate((
                            hidden_states[6][:,0,:].numpy(),
                            torch.gather(hidden_states[6][:,:,:], 1, mask)[:,0,:].numpy(),
                            torch.gather(hidden_states[5][:,:,:], 1, mask)[:,0,:].numpy(),
                            torch.gather(hidden_states[4][:,:,:], 1, mask)[:,0,:].numpy(),
                            torch.gather(hidden_states[3][:,:,:], 1, mask)[:,0,:].numpy()
                           ),
                           axis=1)


# sanity check model output format
assert( np.all(hidden_states[-1].numpy() == last_hidden_states[:,:,:].numpy()) )
labels = batch_1[1]

#Classifier model 
# print("Fit classifier")
# train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

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

