import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from func import loaddata,evaluate,preprocess
import sys
best_net=sys.argv[1]#"model/16_0.400.pkl"
dump=sys.argv[2]#"sample.csv"
sig=nn.Sigmoid()
num_user,num_item,data=loaddata(path="data/train.csv")
#pos_train,pos_val,neg_val=preprocess(num_user,num_item,data)   
#np.save("data/split.npy",[pos_train,pos_val,neg_val])
pos_train,pos_val,neg_val,val=np.load("data/split.npy",allow_pickle=True)

user,item=torch.load(best_net)
pred=sig(torch.matmul(user,item.T))
sort, indices = torch.sort(pred, dim=1,descending=True)
del sort
all_pos=pos_val.union(pos_train)
indices=indices.numpy()
answer=np.empty([num_user,50])
for i in range(num_user):
    ite=0
    for j,ele in enumerate(indices[i]):
        if ite>=50:
            break
        elif (i,ele) not in all_pos:
            answer[i][ite]=ele
            ite+=1
answer=answer.astype("int").astype("str")
file=open(dump,"w")
file.write("UserId,ItemId\n")
for i in range(num_user):
    file.write("%d,%s\n"%(i," ".join(answer[i])))
file.close()
    
