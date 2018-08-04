#Analyze DMC runs with 100 blocks, energy and small RDM without tmoves
import numpy as np 
import json

N=20 #Number of expansions
Ndet=23 #Number of determinants per expansion 

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

#Derivatives
wfnderiv=[]
wfnderiv_err=[]
ederiv=[]
ederiv_err=[]
for cut in ["0p2","0p3","0p4"]:
  for i in range(1,N+1):
    tmp_wfnderiv=[]
    tmp_wfnderiv_err=[]     
    tmp_ederiv=[]
    tmp_ederiv_err=[]     
    basename=el+basis+str(charge)+"_"+method+".dmc_E_SR_"+str(i)+cut+"_not"
    with open("/scratch/sciteam/sapatha2/"+basename+"/"+basename+".o","r") as f:
      for line in f:
        if "Wave function derivatives" in line:
          for j in range(Ndet-1):
	    s=f.next().split(" ")
            tmp_wfnderiv.append(float(s[0]))
	    tmp_wfnderiv_err.append(float(s[2][:-1]))
        if "Derivative of the energy" in line:
          for j in range(N):
	    s=f.next().split(" ")
            tmp_ederiv.append(float(s[0]))
	    tmp_ederiv_err.append(float(s[2][:-1]))
    f.close() 

    wfnderiv.append(tmp_wfnderiv)
    wfnderiv_err.append(tmp_wfnderiv_err)
    ederiv.append(tmp_ederiv)
    ederiv_err.append(tmp_ederiv_err)      

#Ground state
wfnderiv.append([0]*len(wfnderiv[0]))
wfnderiv_err.append([0]*len(wfnderiv[0]))
ederiv.append([0]*len(wfnderiv[0]))
ederiv_err.append([0]*len(wfnderiv[0]))

d={'E':E,'err':err,'wfnderiv':wfnderiv,'wfnderiv_err':wfnderiv_err,
'ederiv':ederiv,'ederiv_err':ederiv_err,'cutoff':cutoff}
json.dump(d,open("analyzedmc_E_SR_not.json","w"))

#Plot energy
'''
import matplotlib.pyplot as plt
import pandas as pd
d=json.load(open("analyzedmc_E_SR_not.json","r"))
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
'''
