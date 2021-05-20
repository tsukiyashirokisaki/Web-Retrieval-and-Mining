import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
sig=nn.Sigmoid()
def loaddata(path):
    file=open(path,"r")
    file=file.read().split("\n")
    data=[]
    for i in range(1,len(file)-1):
        file[i]=file[i].split(",")[1]
        file[i]=file[i].split(" ")
        app=[]
        for ele in file[i]:
            app.append(int(ele))
        data.append(app)
    del file
    maxid=data[0][0]
    minid=data[0][0]
    n=len(data)
    for i in range(n):
        for ele in data[i]:
            maxid=max(ele,maxid)
            minid=min(ele,minid)
    return n,maxid-minid+1,data
def preprocess(num_user,num_item,data):
    #mat=torch.zeros([num_user,num_item])
    pos_val=set()
    pos_train=set()
    neg_val=set()
    for i in range(n):
        m=len(data[i])
        app=set()
        for ele in data[i]:
            #mat[i][ele]+=1
            app.add((i,ele))
        pos_sample=set(random.sample(app,int(m*0.11)))
        neg_sample=set()
        for j in range(int(m*0.11)):
            sam=(i,random.randint(0,maxid))
            if  sam not in app:
                neg_sample.add(sam)
            else:
                j-=1
        neg_val=neg_val.union(neg_sample)
        pos_val=pos_val.union(pos_sample)
        pos_train=pos_train.union(app-pos_sample)
    return pos_train,pos_val,neg_val
def evaluate(user,item,pos_val,neg_val,p,n):
    l=0
    for (j,k) in random.sample(pos_val,p):
        l-=torch.log(sig(torch.matmul(user[j],item[k].view(-1,1))))
    #for (j,k) in random.sample(neg_val,n):
    #    l-=torch.log(1+1e-6-sig(torch.matmul(user[j],item[k].view(-1,1))))
    #l/=(p+n)
    l/=p
    
    return l
    
        
def get_answer(user,item,num_user,exclude):
    pred=sig(torch.matmul(user,item.T))
    sort, indices = torch.sort(pred, dim=1,descending=True)
    del sort
    indices=indices.numpy()
    answer=np.empty([num_user,50])
    for i in range(num_user):
        ite=0
        for j,ele in enumerate(indices[i]):
            if ite>=50:
                break
            elif (i,ele) not in exclude:
                answer[i][ite]=ele
                ite+=1
    answer=answer.astype("int")
    return answer
def MAP(output,pos_val,num_user,val):
    score=0.
    for i in range(num_user):
        s1=0.
        ite=0
        for j in range(50):
            if (i,output[i][j]) in pos_val:
                ite+=1
                s1+=ite/(j+1)
                
        score+=s1/val[i]
        #print(score)
    score/=num_user
    return score