# import libraries
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from keras.preprocessing.sequence import pad_sequences
import torch
import glob

get_ipython().system('pip install natsort')
import natsort
import gc
import random
import os

# set seed
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 42
seed_everything(seed)

# concatenate all the fpc files into one dataframe
f = glob.glob('/fpc-files/*.csv')
sortedfiles = natsort.natsorted(f, reverse=False)

column = ['word']
df = pd.DataFrame()
for files in sortedfiles:
    df = df.append(pd.read_csv(files, usecols=column))

df.reset_index(inplace=True, drop = True)

del f, sortedfiles
gc.collect()


# Optional: Add special tokens to the beginning and the end of a sentence
df.word = df.word.replace({".": ". [SEP]"})
sep_idx = df.index[df['word'] == '. [SEP]'].tolist()

df.word.values[1] = '[CLS] ' + df.word.values[1]

for i in range(0,len(sep_idx)):
    try:
        text = df['word'].values[sep_idx[i] + 2]
        df['word'].values[sep_idx[i] + 2] = '[CLS] ' + text
    except IndexError:
        pass

# Load pre-trained model and tokenizer
config = RobertaConfig.from_pretrained("roberta-large")
config.output_hidden_states = True

rb_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
rb_model = RobertaModel.from_pretrained("roberta-large", config=config)

rb_tokenizer.add_tokens(['[CLS]', '[SEP]'])
rb_model.resize_token_embeddings(len(rb_tokenizer))

def get_ids_masks(df, tokenizer):
    
    input_ids = []
    attention_masks = []

    for sent in df['word']:
        encoded = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = False, 
                            max_length = 8,           
                            pad_to_max_length = True,
                            return_attention_mask = True
                          )

        input_ids.append(encoded.get('input_ids'))
        attention_masks.append(encoded.get('attention_mask'))

    input_ids = torch.LongTensor(input_ids)
    attention_masks = torch.LongTensor(attention_masks)
    
    return input_ids, attention_masks

input_ids, attention_masks = get_ids_masks(df, rb_tokenizer)

# cuda for gpu acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
rb_model.to(device)
rb_model.eval()

with torch.no_grad():
    outputs = rb_model(input_ids.to(device), attention_masks.to(device)) 
    hidden_states = outputs[-1]

token_embeddings = torch.stack(hidden_states, dim=0)
token_embeddings.size()

token_vecs_mean = []
batches = hidden_states[-2]

for batch in batches:
    
    # `batches` is a [1000 x 8 x 1024] tensor
    # `batch` is a [8 x 1024] tensor
    
    # Calculate the average of all 8 token vectors
    mean_vec = torch.mean(batch, dim=0)
    token_vecs_mean.append(mean_vec)

L = [x.cpu().detach().numpy() for x in token_vecs_mean]
arr = np.vstack(L)
emb_df = pd.DataFrame(arr)
emb_df.to_csv("rbl.csv", encoding='utf-8', index = False)


# References:
# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
