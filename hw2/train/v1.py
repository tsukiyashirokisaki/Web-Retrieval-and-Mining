#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import random
import os
import argparse
from func import loaddata,evaluate,preprocess,get_answer,MAP

parser = argparse.ArgumentParser()
parser.add_argument("-lat", type=int,
                    help="num_lat")
parser.add_argument("-m", type=str,
                    help="model")
parser.add_argument("-lr", type=float,
                    help="learning rate",default=1e-1)
parser.add_argument("-r", type=float,
                    help="neg_rate/pos_rate",default=1.)
parser.add_argument("-w", type=float,
                    help="weight_decay",default=1e-5)

args = parser.parse_args()
num_lat=args.lat
sig=nn.Sigmoid()

        
    
    
    
    
num_user,num_item,data=loaddata(path="data/train.csv")
#pos_train,pos_val,neg_val=preprocess(num_user,num_item,data)   
#np.save("data/split.npy",[pos_train,pos_val,neg_val])
pos_train,pos_val,neg_val,val=np.load("data/split.npy",allow_pickle=True)


# In[58]:





# In[92]:



num_ps=100000
num_ns=int(num_ps*args.r)
num_valps=10000
num_valns=10000

train_loss=[]
val_loss=[]
if args.m!=None:
    user,item=torch.load(args.m)
else:
    user=torch.rand(num_user,num_lat,requires_grad=True)
    item=torch.rand(num_item,num_lat,requires_grad=True)
optimizer=torch.optim.Adam([user,item],lr=args.lr,weight_decay=args.w)
best_Map=40
if str(num_lat) not in os.listdir("model"):
    os.mkdir("model/"+str(num_lat))
for i in range(400):
    optimizer.zero_grad()
    loss=0
    for (j,k) in random.sample(pos_train,num_ps):
        loss-=torch.log(sig(torch.matmul(user[j],item[k].view(-1,1))))

    for q in range(num_ns):
        j=random.randint(0,num_user-1)
        k=random.randint(0,num_item-1)
        if (j,k) not in pos_train and (j,k) not in pos_val :#and (j,k) not in neg_val:
            loss-=torch.log(1+1e-6-sig(torch.matmul(user[j],item[k].view(-1,1))))
            #loss-=torch.log(1-sig(torch.matmul(user[j],item[k].view(-1,1))))
        else:
            q-=1
    loss/=(num_ps+num_ns)
    loss.backward()
    optimizer.step()
    #val_loss=evaluate(user,item,pos_val,neg_val,num_valps,num_valns)
    answer=get_answer(user,item,num_user,pos_train)
    Map=MAP(answer,pos_val,num_user,val)*1000
    if Map>best_Map:
        best_Map=Map
        torch.save([user,item],"model/%d/bce/%.3f_%.d_%.d_%.d.pkl"%(num_lat,Map,int(-np.log10(args.r)),int(-np.log10(args.lr)),int(-np.log10(args.w))))
    
    #print("training loss= %.3f validation loss = %.3f MAP=%.3f"%(loss,val_loss,Map))
    print("lat=%d epoch=%d training loss= %.3f MAP=%.3f"%(num_lat,i,loss,Map))



 