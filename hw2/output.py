import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from func import loaddata,evaluate,testprocess
import sys
import os
dump=sys.argv[1]
sig=nn.Sigmoid()
num_user,num_item,data=loaddata(path="train.csv")
all_pos=testprocess(num_user,num_item,data)   
pred=torch.zeros([num_user,num_item])
#best_net=["model/32_0.432.pkl","model/16_0.400.pkl"]
for net in os.listdir("select"):
    user,item=torch.load("select/"+net)
    pred+=sig(torch.matmul(user,item.T))

sort, indices = torch.sort(pred, dim=1,descending=True)
del sort
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
    
