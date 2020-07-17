# import libraries
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
import torch
import glob
import natsort

# concatenate all the fpc files into one dataframe
f = glob.glob('fpc_files/*.csv')
sortedfiles = natsort.natsorted(f, reverse = False)

df = pd.DataFrame()
for files in sortedfiles:
    df = df.append(pd.read_csv(files))

# load pre-trained model and tokenizer
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model = BertModel.from_pretrained(pretrained_weights)

def get_ids_masks(df, tokenizer, model):
    # tokenize the feature column 'word'
    tokenized = df['word'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # pad input text to same length (with the token id 0)
    padded = pad_sequences(tokenized, padding='post')
    
    # create attention masks
    # attend to useful, real tokens only (represented by 1)
    mask = [[int(token_id > 0) for token_id in sentence] for sentence in padded]
    mask = np.asarray(mask)

    # convert data to torch tensors
    input_ids = torch.LongTensor(padded)
    input_mask = torch.LongTensor(mask)
    
    return input_ids, input_mask

input_ids, attention_masks = get_ids_masks(df, tokenizer, model)

# tell pytorch not to compute or store gradients
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_masks)

# layer number, max token number, number of hidden unit / feature 
print(last_hidden_states[0].size())

# extract the last hidden state of the token `[CLS]` 
features = last_hidden_states[0][:,0,:].numpy()

# save as csv file
features_df = pd.DataFrame(features)
features_df.to_csv("bert_emb.csv", encoding='utf-8', index = False)


# Note:
# * An alternative to get the input_ids and attention_masks is to leverage the tokenizer.encode_plus function.
