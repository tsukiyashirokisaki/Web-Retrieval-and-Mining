#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import random
import os
import argparse
from func import loaddata,evaluate,preprocess,get_answer,MAP
sig=nn.Sigmoid()


# In[5]:


num_user,num_item,data=loaddata(path="data/train.csv")
#pos_train,pos_val,neg_val=preprocess(num_user,num_item,data)   
#np.save("data/split.npy",[pos_train,pos_val,neg_val])
pos_train,pos_val,neg_val,val=np.load("data/split.npy",allow_pickle=True)


# In[8]:



num_ps=50000
#num_ns=50000
num_valps=10000
num_valns=10000
for num_lat in [16,32,64,128]:
    user=torch.rand(num_user,num_lat,requires_grad=True)
    item=torch.rand(num_item,num_lat,requires_grad=True)
    optimizer=torch.optim.Adam([user,item],lr=1e-1,weight_decay=1e-4)
    best_Map=10
    for i in range(1000):
        optimizer.zero_grad()
        loss=0
        for (j,k) in random.sample(pos_train,num_ps):
            #loss-=torch.log(sig(torch.matmul(user[j],item[k].view(-1,1))))
            l=random.randint(0,num_item-1)
            while (j,l) in pos_train or (j,l) in pos_val:
                l= random.randint(0,num_item-1)
            loss+=torch.log(1e-6+sig(torch.matmul(user[j],item[k].view(-1,1))-torch.matmul(user[j],item[l].view(-1,1))))

        
        loss/=num_ps
        
        loss.backward()
        answer=get_answer(user,item,num_user,pos_train)
        Map=MAP(answer,pos_val,num_user,val)*1000
        if Map>best_Map:
            best_Map=Map
            torch.save([user,item],"model/%d/bpr/%.3f_%.2f_%.d.pkl"%(num_lat,Map,args.r,int(-np.log10(args.lr))))

        #print("training loss= %.3f validation loss = %.3f MAP=%.3f"%(loss,val_loss,Map))
        print("lat=%d epoch=%d training loss= %.3f MAP=%.3f"%(num_lat,i,loss,Map))
