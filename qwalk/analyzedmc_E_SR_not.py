#Analyze DMC runs with 100 blocks, energy and small RDM without tmoves
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
cutoff=[]
for cut in ["0p2","0p3","0p4"]:
  for i in range(1,N+1):
    e=[]
    esig=[]
    nb=0
    basename=el+basis+str(charge)+"_"+method+".dmc_E_SR_"+str(i)+cut+"_not"
    with open("/scratch/sciteam/sapatha2/"+basename+"/"+basename+".o","r") as f:
      for line in f:
        if "total_energy0" in line:
	  E.append(-1*float(line.split("-")[1][:-3]))
	  err.append(float(line.split("-")[2].split("(")[0]))
          if(cut=="0p2"): cutoff.append(0.2)
	  elif(cut=="0p3"): cutoff.append(0.3)
	  else: cutoff.append(0.4)
    f.close() 

#Ground state
E.append(-213.4851715)
err.append(0.0005193355127)
cutoff.append(0)
assert(min(E)==-213.4851715)

d={'E':E,'err':err,'cutoff':cutoff}
json.dump(d,open("analyzedmc_E_SR_not.json","w"))

'''
import matplotlib.pyplot as plt 
d=json.load(open("analyzedmc.json","r"))
d['E']=np.array(d['E'])
d['err']=np.array(d['err'])
ind=np.argsort(d['E'])
d['E']=np.sort(d['E'])
d['E']-=d['E'][0]
d['err']=d['err'][ind]
d['E']*=27.2
d['err']*=27.2
plt.errorbar(np.arange(len(d['E'])),d['E'],yerr=d['err'],c='g',marker='o')
plt.show()
'''
