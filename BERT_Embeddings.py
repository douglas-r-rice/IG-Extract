# import libraries
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
import torch

# load data
data = pd.read_csv("fpc_all.csv")

# load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

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
    input_ids = torch.tensor(padded)
    input_mask = torch.tensor(mask)
    
    return input_ids, input_mask

input_ids, attention_masks = get_ids_masks(data, tokenizer, model)

with torch.no_grad():
    last_hidden_states = model(input_ids.long(), attention_masks)

# layer number, max token number, number of hidden unit / feature 
print(last_hidden_states[0].size())

features = last_hidden_states[0][:,0,:].numpy()

# save as csv file
features_df = pd.DataFrame(features)
features_df.to_csv("feature_embeddings.csv", encoding='utf-8', index = False)


# Notes:
# * To concatenate all the fpc files into one dataframe, check out this [thread](https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe) on Stack Overflow.
# * An alternative to using the ```get_ids_masks``` function to get the input_ids and attention_masks is to use ```tokenizer.encode_plus```. The function has the ability to do everything listed in the ```get_ids_masks``` function. Check out the [documentation](https://huggingface.co/transformers/main_classes/tokenizer.html?highlight=encode_plus#transformers.PreTrainedTokenizer.encode_plus).
