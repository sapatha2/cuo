#Generate distribution of cutoff data 
import numpy as np 
import matplotlib.pyplot as plt 
import json 

cutoff=0.5
fname="Cuvtz0_B3LYP_s3_g0.1_c"+str(cutoff)+".vmc.out"
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
        else: 
          dropped[j].append(float(l))
          j+=1
        
    
json.dump(dropped,open(fname+".json","w"))     
