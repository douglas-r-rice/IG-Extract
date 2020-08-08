# import libraries
import pandas as pd
import numpy as np
import glob

get_ipython().system('pip install natsort')
import natsort

# concatenate all the fpc files into one dataframe
f = glob.glob('/fpc-files/*.csv')
sortedfiles = natsort.natsorted(f, reverse=False)

df = pd.DataFrame()
for files in sortedfiles:
    df = df.append(pd.read_csv(files))

get_ipython().system('pip install tensorflow==1.15')
# !pip install bert-serving-client
# !pip install -U bert-serving-server[http]
get_ipython().system('pip install -U bert-serving-server bert-serving-client')
get_ipython().system('wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')
get_ipython().system('unzip uncased_L-12_H-768_A-12.zip')

import subprocess
from bert_serving.client import BertClient
bert_command = 'bert-serving-start -model_dir /uncased_L-12_H-768_A-12/ -num_worker=4 -max_batch_size=256 -max_seq_len=9'
process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)
bc = BertClient()

words = df['word'].to_list()

encoded = bc.encode(words)
features = pd.DataFrame(encoded)

features.to_csv("bert_service_emb.csv", encoding='utf-8', index = False)

# Reference:
# https://github.com/hanxiao/bert-as-service
