
# coding: utf-8

# # General-domain LM pretraining

# In[1]:


import json
import pathlib

from fastai.text import *

import numpy as np
import pandas as pd
import html


# In[2]:


BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

PATH = pathlib.Path("lm/pt/data/")


# In[3]:


LM_PATH=Path('lm/pt/pt_lm/')
LM_PATH.mkdir(exist_ok=True)


# ## Loading dataset 

# In[4]:


tok_trn = np.load(LM_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(LM_PATH/'tmp'/'tok_val.npy')


# In[5]:


# Identify the most common tokens and numericalizing the text
freq = Counter(p for o in tok_trn for p in o) 
freq.most_common(25)


# In[6]:


em_sz,nh,nl = 400,1150,3
wd=1e-7
bptt=70
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))


# ### 1-cycle 

# In[7]:


# Truncating our vocab to ignore the rare words
max_vocab = 30000
min_freq = 5

itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq] # getting rid of the rare words
itos.insert(0, '_pad_') # 
itos.insert(0, '_unk_') # itos is the list of all the strings in the vocab


# In[8]:


# creating a index-key dictionary for our vocabulary
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)


# In[9]:


# creating a index representation for our train and validation dataset
trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])


# In[10]:


# saving our indexed representation of our dataset to disk
# we also save the index-word mapping to retrieve the complete text representation from these numpy arrays
np.save(LM_PATH/'tmp'/'trn_ids.npy', trn_lm)
np.save(LM_PATH/'tmp'/'val_ids.npy', val_lm)
pickle.dump(itos, open(LM_PATH/'tmp'/'itos.pkl', 'wb'))


# In[11]:


# Loading the indexed representation of our dataset from disk
# we also load the index-word mapping to to help us convert the indexes to word datasets, if need be.
trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))


# In[12]:


# checking vocabulary size
vs=len(itos)
vs,len(trn_lm)


# In[13]:


bs = 52


# In[14]:


trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)


# In[15]:


opt_fn = partial(optim.SGD, momentum=0.9)


# In[16]:


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.05 #the higher the lr, the lower dp


# In[17]:


learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])


# In[18]:


learner.load('lm_PT_1_cycle')


# In[19]:


learner.metrics = [accuracy]
learner.unfreeze()


# In[20]:


lr = 5


# In[21]:


learner.fit(lr, 1, cycle_len=10, use_clr_beta=(10,10,0.95,0.85))


# In[ ]:


learner.save('lm_PT_1_cycle_10_epochs')


# In[ ]:


learner.save_encoder('lm_PT_1_cycle_10_epochs_enc')


