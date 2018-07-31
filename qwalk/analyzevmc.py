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
for i in range(1,N+1):
  e=[]
  esig=[]
  basename=el+basis+str(charge)+"_"+method+".vmc"+str(i)
  with open("/scratch/sciteam/sapatha2/"+basename+"/"+basename+".o","r") as f:
    for line in f:
      if("##0" in line):
	e.append(float(line.split(" ")[10]))	
      if("&&0" in line):
	esig.append(float(line.split(" ")[11]))	
  e=np.array(e)
  esig=np.array(esig)
  E.append(np.sum(e)/len(e))
  err.append(np.sqrt(np.sum(esig**2)/len(e)**2))
  f.close() 


d={'E':E,'err':err,'nblock':len(e)}
json.dump(d,open("analyzevmc.json","w"))
