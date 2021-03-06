#Analyze DMC runs with 400 blocks, energy only with tmoves
import numpy as np 
import json

N=20 #Number of expansions 

el='Cu'
charge=0
minao={}
basis='vtz'
method='B3LYP'

'''
#Energies
E=[]
err=[]
cutoff=[]
for cut in ["0p2","","0p4"]:
  for i in range(1,N+1):
    e=[]
    esig=[]
    nb=0
    basename=el+basis+str(charge)+"_"+method+".dmc_E"+str(i)+cut
    with open("/scratch/sciteam/sapatha2/"+basename+"/"+basename+".o","r") as f:
      for line in f:
        if "total_energy0" in line:
	  E.append(-1*float(line.split("-")[1][:-3]))
	  err.append(float(line.split("-")[2].split("(")[0]))
          if(cut=="0p2"): cutoff.append(0.2)
	  elif(cut==""): cutoff.append(0.3)
	  else: cutoff.append(0.4)
    f.close() 

#Ground state
E.append(-213.4851715)
err.append(0.0005193355127)
cutoff.append(0)
assert(min(E)==-213.4851715)

d={'E':E,'err':err,'cutoff':cutoff}
json.dump(d,open("analyzedmc.json","w"))
'''

import matplotlib.pyplot as plt 
import pandas as pd
d=json.load(open("analyzedmc.json","r"))
df=pd.DataFrame(d)
for cutoff in [0.4,0.3,0.2]:
  #Select values for cutoff
  sel=(np.abs(cutoff-(np.array(d['cutoff'])))<1e-5)
  E=np.array(d['E'])[sel]
  err=np.array(d['err'])[sel]
  #Add ground state energy
  E=np.insert(E,0,d['E'][-1])
  err=np.insert(err,0,d['err'][-1])
  #Sort energies
  ind=np.argsort(E)
  E=E[ind]
  err=err[ind]
  E-=E[0]  #Subtract GS energy
  E*=27.2
  err*=27.2
  plt.errorbar(np.arange(len(E)),E,yerr=err,marker='o',label='cutoff='+str(cutoff))

plt.title("Total DMC energies CuO")
plt.ylabel("E-E[GS], eV")
plt.xlabel("State")
plt.legend(loc=2)
plt.show()
