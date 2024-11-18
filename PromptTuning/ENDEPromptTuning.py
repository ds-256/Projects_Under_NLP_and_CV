# -*- coding: utf-8 -*-
"""prompttuning(2).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sSSuV-EjMw171QSg1mJSiFoHEfWWrsD5
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import torch
from torch import tensor
import torch.nn as nn
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset,DataLoader
import math

device='cuda' if torch.cuda.is_available() else 'cpu'

!pip install transformers

#train_Data=pd.read_csv('/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv')
#val_Data=pd.read_csv('/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/validation.csv')
#test_Data=pd.read_csv('/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/test.csv')

train_Data=pd.read_csv('/kaggle/input/europarl-parallel-corpus-19962011/english_german.csv')
train_Data=train_Data[:9000]
#val_Data=pd.read_csv('/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/validation.csv')
test_Data=pd.read_csv('/kaggle/input/europarl-parallel-corpus-19962011/english_german.csv')[40000:52000]

len(train_Data)

train_Data['new']='[TRANSLATE_EN_DE] '+train_Data[train_Data.columns[0]]+' '+train_Data[train_Data.columns[1]]

test_Data['new']='[TRANSLATE_EN_DE] '+test_Data[train_Data.columns[0]]+' '+test_Data[test_Data.columns[1]]

train_Data=train_Data.drop(train_Data.columns[0],axis=1)
test_Data=test_Data.drop(test_Data.columns[0],axis=1)



from transformers import GPT2Config,GPT2LMHeadModel

configuration=GPT2Config(n_positions=768)
model=GPT2LMHeadModel(configuration)
configuration=model.config

torch.tensor(tokenizer.encode(train_Data[train_Data.columns[1]][5])).shape

for params in model.named_parameters():
    if params[0]=='wte.weight':
        params[1].requires_grad=False
        #print(params[1][:20])
    else:
        params[1].requires_grad=False
    #params.requires_grad=False

model.get_input_embeddings()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model)

configuration.n_positions

from transformers import GPT2TokenizerFast
tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token':tokenizer.eos_token})

train_Data[train_Data.columns[1]]

train_Data[train_Data.columns[0]]=train_Data[train_Data.columns[0]].astype(str)
train_Data[train_Data.columns[1]]=train_Data[train_Data.columns[1]].astype(str)

test_Data[test_Data.columns[0]]=test_Data[test_Data.columns[0]].astype(str)
test_Data[test_Data.columns[1]]=test_Data[test_Data.columns[1]].astype(str)

class DataFetch(Dataset):
    def __init__(self,split:str):
        data=[]
        labels=[]
        for x in tqdm(split[split.columns[0]],desc='Generate  data'):
            if len(x)<2 or x[0]=='':continue
            words=x
            data.append(x)
        for x in tqdm(split[split.columns[1]],desc='Generate Labels'):
            if len(x)<2 or x[0]=='':continue
            words=x
            #indices.append(vocab['<EOS>'])
            labels.append(x)
        self.data=data
        self.labels=labels
    def __len__(self)->int:
        return len(self.data)
    def __getitem__(self,index:int):
        return self.data[index],self.labels[index]



train_Data=train_Data[:-2]

len(train_Data)

df=DataFetch(train_Data)
train_dataloader=DataLoader(df,batch_size=6,shuffle=True)

df_test=DataFetch(test_Data[:-2])
test_dataloader=DataLoader(df_test,batch_size=6,shuffle=True)

d,l=next(iter(train_dataloader))

for x in test_dataloader:
    l,b=x
    print(len(l))
    break

z1=torch.zeros((700,768))
z2=torch.zeros((68,768))
z3=torch.cat([z1,z2],axis=0)

z3.shape

words='[SUMMARIZE] [QUESTION] [ANSWER] [TRANSLATE_EN_DE]'
vocab=build_vocab_from_iterator(words,min_freq=1,specials=['<pad>'])
vocab.set_default_index(vocab.get_stoi()[' '])

len(vocab)

class PromptTuning(nn.Module):
    def __init__(self,gpt2model,tokenizer,task_type,n_tokens=68,random_range=0.5):
        super(PromptTuning,self).__init__()
        self.gpt2model=gpt2model
        self.gpt2model.requires_grad=False
        self.n_tokens=n_tokens
        self.random_range=random_range
        self.tokenizer=tokenizer
        self.task_type=task_type
        self.emb1=nn.Embedding(n_tokens,768)
        self.emb=nn.parameter.Parameter(self.initialize_emb(self.emb1,task_type,n_tokens))
    def initialize_emb(self,emb,task_type,n_tokens):
        if task_type=='s':
            l=[]
            for k in '[SUMMARIZE]':
                l.append(vocab.lookup_indices([k]))
            z=emb(torch.tensor(l)).squeeze(1)
            extra_rem=n_tokens-z.shape[0]
            z1=torch.FloatTensor(emb.weight[:extra_rem])
            return torch.cat([z,z1],dim=0)
        elif task_type=='qa':
            l=[]
            for k in '[QUESTION], [ANSWER]':
                l.append(vocab.lookup_indices([k]))
            z=emb(torch.tensor(l)).squeeze(1)
            extra_rem=n_tokens-z.shape[0]
            z1=torch.FloatTensor(emb.weight[:extra_rem])
            return torch.cat([z,z1],dim=0)
        elif task_type=='mt':
            l=[]
            for k in '[TRANSLATE_EN_DE]':
                l.append(vocab.lookup_indices([k]))
            z=emb(torch.tensor(l)).squeeze(1)
            extra_rem=n_tokens-z.shape[0]
            z1=torch.FloatTensor(emb.weight[:extra_rem])
            return torch.cat([z,z1],dim=0)
    def forward(self,inp):
        task_specific=self.emb
        #print(task_specific.shape,tokenized_ip.shape)
        inp_embedding=model.transformer.wte.weight[inp]
        #print(inp_embedding.shape,task_specific.shape)
        final_emb=torch.cat([task_specific[None,:,:].repeat(6,1,1),inp_embedding],dim=1).to(device)
        return model(inputs_embeds=final_emb)

pt=PromptTuning(model,tokenizer,'mt',68,0.5)

pt_opt=torch.optim.Adam(pt.parameters(),lr=1e-2)
loss_fn=torch.nn.CrossEntropyLoss()

del pt
del labels_id
del data_id
del out
del loss

import gc
gc.collect()
torch.cuda.empty_cache()
#torch.cuda.memory_allocated()
#model.cpu()
#del pt
gc.collect()
torch.cuda.empty_cache()

torch.cuda.memory_allocated()

pt.to(device)

for params in pt.parameters():
    if params.requires_grad==True:
        print(params.shape)

pt.train()
epoch_loss_main=[]
for epoch in range(1):
    epoch_loss=0
    for batch in tqdm(train_dataloader,desc='training'):
        pt_opt.zero_grad()
        data,labels=batch
        labels_id=tokenizer.batch_encode_plus(labels,return_tensors='pt',truncation=True,max_length=768,padding='max_length')['input_ids']
        data_id=tokenized_ip=tokenizer.batch_encode_plus(data,return_tensors='pt',truncation=True,max_length=700,padding='max_length')['input_ids']
        #print(labels_id.shape)
        out=pt(data_id)
        loss=loss_fn(out[0].view(out[0].shape[0]*out[0].shape[1],out[0].shape[2]),labels_id.view(labels_id.shape[0]*labels_id.shape[1]).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pt.parameters(),1.0)
        pt_opt.step()
        epoch_loss+=loss.item()
    epoch_loss_main.append(epoch_loss)

torch.save(pt,'/kaggle/working/EnDETranslation')

pt=torch.load('/kaggle/working/EnDETranslation')

!pip install evaluate
!pip install rouge_score
import evaluate
rouge=evaluate.load("rouge")

from torchtext.data.metrics import bleu_score

bleu=evaluate.load("bleu")

pt.eval()
epoch_loss=0
rouge_test=[]
bleu_score_=[]
for batch in tqdm(test_dataloader,desc='testing'):
    rougue_batch=[]
    bleu_batch=[]
    data,labels=batch
    labels_id=tokenizer.batch_encode_plus(labels,return_tensors='pt',truncation=True,max_length=768,padding='max_length')['input_ids'].to(device)
    data_id=tokenized_ip=tokenizer.batch_encode_plus(data,return_tensors='pt',truncation=True,max_length=700,padding='max_length')['input_ids'].to(device)
    #print(labels_id.shape)
    out=pt(data_id.to(device))
    out_max=torch.argmax(out[0],dim=2)
    for k in range(6):
        z=[tokenizer.decode(out_max[k])]
        rougue_batch.append(rouge.compute(predictions=z,references=[labels[k]]))
        z1=[tokenizer.decode(out_max[k],skip_special_tokens=True,clean_up_tokenization_spaces=True)]
        bleu_batch.append(bleu.compute(predictions=z1,references=[labels[k]]))
    bleu_score_.append(bleu_batch)
    rouge_test.append(rougue_batch)

bleu.compute(predictions=[tokenizer.decode(out_max[0],skip_special_tokens=True,clean_up_tokenization_spaces=True)],references=[labels[0]])

with open('/kaggle/working/bleu.txt','w') as f:
    for i in bleu_score_:
        f.write(str(i))

with open('/kaggle/working/rougue2.txt','w') as f:
    for i in rouge_test:
        f.write(str(i))
