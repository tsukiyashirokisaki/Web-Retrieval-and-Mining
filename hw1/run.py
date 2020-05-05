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
from func_set import build_query_dic,build_doc_vector,build_text,build_doc_vector,output,build_term_dic
if_train=int(sys.argv[1])
model_dir="model"
if if_train:
	query_dir="queries/query-train.xml"
	num=10
else:
	#query_dir="queries/query-train.xml"
	#num=10
	query_dir="queries/query-test.xml"
	num=20

###Import DOC
def main(k1,b,k3,feature,query_feature,r):
	file_list=open(model_dir+"/file-list")
	file=file_list.read().split("\n")
	file.remove("")
	num_doc=len(file) 

	file_dic=dict()
	for i in range(num_doc):
	    term=file[i].split("/")[-1].lower()
	    file_dic[term]=i

	inf_file=open(model_dir+"/inverted-file")
	inf=inf_file.read()
	del inf_file
	inf=inf.split("\n")
	inf.remove("")

	dictionaries,que_dic,select_voc,num_term,que=build_term_dic(model_dir,query_dir,num,k3=k3,F=
		feature,QF=query_feature)
	D=build_doc_vector(num_doc,dictionaries,inf,select_voc,k1=k1,b=b)
	#pca = PCA(n_components=int(num_term*0.9), svd_solver='full')
	#D_=pca.fit_transform(D)
	D_=D
	if if_train:
		df=pd.read_csv("queries/ans_train.csv")
		truth=[]
		for i in range(num):
		    app=[]
		    ret=df["retrieved_docs"][i].split(" ")
		    for ele in ret:
		        app.append(file_dic[ele])
		    truth.append(app)
		w_file=open("score.txt","a")
		for i in r:
			train_rank=output(que,num,num_doc,D_,r=i)#,feature=["title","concepts","question"])
			
			for j in range(num):
				print(ml_metrics.apk(truth[j], train_rank[j,:100].tolist(), k=100))
			score=ml_metrics.mapk(truth, train_rank[:,:100].tolist(), k=100)
			#w_file.write("s=%.3f,k1=%.1f, k3=%d, b=%.2f ,r=%d, f="%(score,k1,k3,b,i)+str(feature)+" q="+str(query_feature)+"\n")
			print(i,score)
		w_file.close()

	else:
		
		train_rank=output(que,num,num_doc,D_,r=r)[:,:100].tolist()
		return train_rank,file
if if_train:
	grid=np.load("grid.npy",allow_pickle=1).tolist()
	#r=[0,1,2,3,4]
	k1=1.2
	k3=100
	b=0.65
	r=[0]
	feature=[1]
	query_feature=[1]
	main(k1,b,k3,feature,query_feature,r)
	#for feature in [[0,1,2,3]]:
	#	for query_feature in [[1,2]]:
	#		for k1 in np.arange(1.2,2.1,0.1):
	#			for k3 in np.arange(100,1100,100):
	#				for b in np.arange(0.5,1.5,0.05):
	#					main(k1,b,k3,feature,query_feature,r)
else:
	k1=1.2
	k3=100
	b=0.65
	r=0
	feature=[1]
	query_feature=[1]
	name="concept_only"
	train_rank,file=main(k1,b,k3,feature,query_feature,r)
	w=open("%s.csv"%(name),"w")
	w.write("query_id,retrieved_docs\n")
	for i in range(num):
		w.write("0%d,"%(i+11))
		for ele in train_rank[i]:
			w.write(file[int(ele)].split("/")[-1].lower()+" ")
		w.write("\n")
	w.close()
