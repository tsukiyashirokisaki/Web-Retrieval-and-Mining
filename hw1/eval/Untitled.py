
def create_grid(num):
    grid=[]
    for i in range(num):
        for j in range(num):
            if i==0:
                grid.append([j])
            elif i==1:
                grid.append([i,j])
            elif i==2:
                grid.append([(j+1)%num,(j+2)%num,(j+3)%num])
                
    return grid

def create1(num):
    return [[ele] for ele in range(num)]
def create2(num):
    app=[]
    for i in range(num):
        for j in range(i+1,num):
            app.append([i,j])
    return app
def create3(num):
    app=[]
    for i in range(num):
        for j in range(i+1,num):
            for k in range(j+1,num):
                app.append([i,j,k])
    return app
#grid=create1(4)+create2(4)+create3(4)


#import numpy as np
#np.save("grid.npy",grid+[[1,2,3,4]])

#np.load("grid.npy",allow_pickle=1).tolist()[5]

def MAP(truth,predict):
    score=0.0
    n=len(predict)
    for i in range(n):
        app_score=0.0
        ite=1
        for j,ele in enumerate(predict[i]):
            if ele in truth[i]:
                app_score+=ite/(j+1)
                ite+=1
        app_score/=len(truth[i])
        score+=app_score
    score/=len(predict)
    return score


import pandas as pd
def readans(path):
    df=pd.read_csv(path)
    truth=[]
    for i in range(len(df)):
        app=[]
        ret=df["retrieved_docs"][i].split(" ")
        for ele in ret:
            app.append(ele)
        truth.append(app)
    return truth


import ml_metrics 
import sys
truth=readans("ans_train.csv")
predict=readans("%s.csv"%(sys.argv[1]))
print("overall score",ml_metrics.mapk(truth, predict, k=100))

for i in range(len(truth)):
    print(i,len(truth[i]),ml_metrics.apk(truth[i], predict[i], k=100))
