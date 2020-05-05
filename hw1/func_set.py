import sys
import numpy as np
import pandas as pd
import re
import xml.etree.ElementTree as ET
import pandas as pd
import ml_metrics 
import pickle as pk
from sklearn.decomposition import PCA
import jieba
def build_query_dic(path):
    stopwords=[];segments=[]
    with open('stopwords.txt', 'r', encoding='UTF-8') as stop_file:
        for data in stop_file.readlines():
            data = data.strip()
            stopwords.append(data)

    tree = ET.parse(path)

    tag_lis=["title","concepts","question","narrative"]
    que_dic=dict()
    for ele in tag_lis:
        que_dic[ele]=[]
        for elem in tree.iter(tag=ele):
            s=elem.text
            if ele=="concepts":
                s=s.replace("。","")
                s=s.replace("\n","")            
                s=s.replace(" ","")                
                s=s.split("、")
                app=[]
                for i in range(len(s)):
                    app.append(s[i])
                que_dic[ele].append(app)
            else:
                for stop in stopwords:
                    s=s.replace(stop,"")
                s=s.replace("\n","")
                que_dic[ele].append(s)

    return que_dic
def build_text(feature,que_dic,num):
    q_doc=[]
    for i in range(num):
        s=""
        for tag in feature:
            s+=que_dic[tag][i]    
        q_doc.append(s)
    return q_doc

def build_term_dic(model_dir,query_dir,num,k3=100,F=[0,1,2],QF=[0,1,2]):
    tag_lis=["title","concepts","question","narrative"]
    feature=[]
    query_feature=[]    
    for i in range(len(F)):
        feature.append(tag_lis[F[i]])
    query_feature=[]
    for i in range(len( QF)):
        query_feature.append(tag_lis[QF[i]])
    que_dic=build_query_dic(query_dir)
    vocab=open(model_dir+"/vocab.all","r",encoding="utf8")
    ind2voc=vocab.read().split("\n")
    del vocab
    voc2ind=dict()
    for i,ele in enumerate(ind2voc):
        voc2ind[ele]=i
    ind2term=[]
    for  i in range(len(feature)):
        for ele in que_dic[feature[i]]:
            ind2term+=ele
    i=0
    while i<len(ind2term):
        k=ind2term[i]
        n=len(k)
        if n==1:
            ind2term.remove(k)
            i-=1
        elif n>2:
            for j in range(n-1):
                ind2term.append(k[j]+k[j+1])            
            ind2term.remove(k)
            i-=1

        i+=1
    ind2term=list(set(ind2term))
    term2ind=dict()
    for i,ele in enumerate(ind2term):
        term2ind[ele]=i

    select_voc=[]
    for i in range(len(ind2term)):
        select_voc.append(voc2ind[ind2term[i][0]])      
    select_voc=list(set(select_voc))    
    num_term=len(term2ind)
    dictionaries=[voc2ind,ind2voc,term2ind,ind2term]

    #q_doc=build_text(query_feature,que_dic,num)
    q_doc=[]
    for i in range(num):
        q_doc.append([])
    for j in range(len(query_feature)):
        for i in range(num):
            q_doc[i]+=que_dic[query_feature[j]][i]
    que=np.zeros([num,num_term])
    #segments=[];term=[]
    for i in range(num):
        #segments =  jieba.cut_for_search(q_doc[i])
        #term = list(filter(lambda a: a not in stopwords and a != '\n', segments))
        term=q_doc[i]
        for ele in term:
            n=len(ele)
            if n==2 and ele in term2ind:
                que[i][term2ind[ele]]+=1
                #print(ele)
            
                
            elif n>2:
                for j in range(n-1):
                    if ele[j]+ele[j+1] in term2ind:
                        #print(ele)
                        que[i][term2ind[ele[j]+ele[j+1]]]+=1
                        #print(ele[j]+ele[j+1])
                
    for i in range(num):
        for j in range(num_term):
            que[i,j]=(k3+1)*que[i,j]/(k3+que[i,j])
    return dictionaries,que_dic,select_voc,num_term,que
def build_doc_vector(num_doc,dictionaries,inf,select_voc,k1=1.2,b=0.65):# k1=1.2-2.0 and b=0.75
    voc2ind,ind2voc,term2ind,ind2term=dictionaries
    num_term=len(ind2term)
    dl=np.zeros([num_doc,1])
    D=np.zeros([num_doc,num_term])
    IDF=np.zeros([num_term])
    i=0
    while i<len(inf):
        prep=[int(ele) for ele in inf[i].split(" ")]
        n=len(prep)
        if n==3:
            voc1,voc2,df=prep
            if voc2==-1:
                i+=df
            elif voc1 in select_voc:
                term=ind2voc[voc1]+ind2voc[voc2]
                if  term in term2ind:
                    ind=term2ind[term]
                    IDF[ind]=np.log((num_doc-df+0.5)/(df+0.5))
                else:
                    i+=df
            else:
                i+=df
        elif n==2:
            doc_ind,TF=prep
            D[doc_ind,ind]=TF
        i+=1
    dl=np.sum(D,axis=1)
    avg_dl=np.mean(dl)
    for i in range(num_doc):
        for j in range(num_term):
            D[i,j]=IDF[ind]*D[i,j]*(k1+1)/(D[i,j]+k1*(1-b+b*dl[doc_ind]/avg_dl))
    return D
def output(que,num,num_doc,D_,r=0,rel=20,alpha=1,beta=1,gamma=0.5):

    
    #que=pca.transform(que)
    result=np.matmul(que,D_.T)
    doc_rank=np.empty([num,num_doc])
    for i in range(num):
        s=result[i]
        doc_rank[i]=sorted(range(len(s)), key=lambda k: s[k],reverse=1)
    for k in range(r):
        que=alpha*que
        not_rel=num-rel
        for i in range(num):
            for j in range(rel):
                que[i]+=D_[int(doc_rank[i,j])]*beta/float(rel)
            for j in range(rel,num):
                que[i]-=D_[int(doc_rank[i,j])]*gamma/float(not_rel)
        result=np.matmul(que,D_.T)
        doc_rank=np.empty([num,num_doc])
        for i in range(num):
            s=result[i]
            doc_rank[i]=sorted(range(len(s)), key=lambda k: s[k],reverse=1)
    return doc_rank

def ret_doc(path):
    tree = ET.parse(path)
    s=""
    for elem in tree.iter(tag='title'):
        if elem.text is not None:
            s+=elem.text
    
    for elem in tree.iter(tag='p'):
        s+=elem.text
    s=s.replace("\n","")
    return s
