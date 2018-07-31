#Analyze VMC
import numpy as np 
import json
N=20 #Number of expansions 

el='Cu'
charge=0
minao={}
basis='vtz'
method='B3LYP'

#Energies
E=[]
err=[]
nblock=[]
for i in range(1,N+1):
  e=[]
  esig=[]
  nb=0
  basename=el+basis+str(charge)+"_"+method+".vmc"+str(i)
  with open("/scratch/sciteam/sapatha2/"+basename+"/"+basename+".o","r") as f:
    for line in f:
      if("##0" in line):
	e.append(float(line.split(" ")[10]))	
        nb+=1
      if("&&0" in line):
	esig.append(float(line.split(" ")[11]))	
  e=np.array(e)
  esig=np.array(esig)
   
  if(i==5): print(esig)

  e=np.sum(e)/len(e)
  esig=np.sum(esig**2)/len(esig)**2  
  E.append(e)
  err.append(esig)
  nblock.append(nb)
  print(e,esig)
  f.close() 


d={'E':E,'err':err,'nblock':nblock}
json.dump(d,open("analyzevmc.json","w"))
