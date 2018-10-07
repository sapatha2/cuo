#Generate distribution of cutoff data 
import sys
sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
from analyze_jsonlog import gather_json_df
import numpy as np 
import matplotlib.pyplot as plt 
import json 
from scipy.stats import probplot

cutoff=0.75
fname="Cuvtz0_B3LYP_s3_g0.1_c"+str(cutoff)+".vmc.out"
'''
dropped=[[],[],[],[],[],[],[],[],[],[]]

with open(fname,"r") as f:
  for line in f:
    if "Drop" in line:
      j=0
      for i in range(15):
        l=next(f)
        if(j==10):
          break
        elif "," in l: 
          pass
        elif "BIG" in l:
          pass
        elif "Drop" in l:
          pass
        else: 
          dropped[j].append(float(l))
          j+=1
        
    
json.dump(dropped,open(fname+".json","w"))     
'''
dropped=json.load(open(fname+".json","r"))

#Plot QQ or something 
fname='Cuvtz0_B3LYP_s3_g0.1_c'+str(cutoff)+'.vmc.json'
d=gather_json_df(fname)
data=list(d['dpwf_1'])
data+=dropped[1]
print(len(data))
res = probplot(data, plot=plt)
plt.show()
