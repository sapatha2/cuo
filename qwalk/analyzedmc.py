#Analyze DMC 
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
  nb=0
  basename=el+basis+str(charge)+"_"+method+".dmc_E"+str(i)
  with open("/scratch/sciteam/sapatha2/"+basename+"/"+basename+".o","r") as f:
    for line in f:
      if "total_energy0" in line:
	E.append(float(line.split("-")[1][:-3]))
	err.append(float(line.split("-")[2].split("(")[0]))
  f.close() 

d={'E':E,'err':err}
json.dump(d,open("analyzedmc.json","w"))
