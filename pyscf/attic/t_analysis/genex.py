import numpy as np 
import matplotlib.pyplot as plt
from methods import genex
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
from functools import reduce
import pandas as pd 
import seaborn as sns 
import statsmodels.api as sm 
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.model_selection import cross_val_score

f='b3lyp_mo_symm.pickle'
a=np.load(f)

r=1.963925
method='B3LYP'
basis='vtz'
el='Cu'
charge=0

'''
ncore={'-1':5,
        '1':5,
        '3':5}
nact={'-1':[8,7],
       '1':[8,7],
       '3':[9,6]}
act={'-1':[np.arange(5,14),np.arange(5,14)],
      '1':[np.arange(5,14),np.arange(5,14)],
      '3':[np.arange(5,14),np.arange(5,14)]}
'''
ncore={'-1':9,
        '1':[0,1,2,3,4,8,9],
        '3':[0,1,2,3,4,8,9]}
nact={'-1':[6,5],
       '1':[6,5],
       '3':[7,4]}
act={'-1':[np.arange(5,14),np.arange(5,14)],
      '1':[[5,6,7,10,11,12,13],[5,6,7,10,11,12,13]],
      '3':[[5,6,7,10,11,12,13],[5,6,7,10,11,12,13]]}

data=None
for mol_spin in [1,3]:
  chkfile="../chkfiles/"+el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method+".chk"
  mol=lib.chkfile.load_mol(chkfile)
  if("U" in method): m=UHF(mol)
  else: m=ROHF(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

  ##################
  #BUILD EXCITATIONS 
  mo_occ=np.array([np.ceil(m.mo_occ-m.mo_occ/2),np.floor(m.mo_occ/2)])
  detgen='a'
  N=100
  Ndet=10
  c=0.8
  beta=0 #Beta for weights
  st=str(mol_spin)
  dm_list=genex(mo_occ,m.mo_energy,ncore[st],act[st],nact[st],N,Ndet,detgen,c,beta)

  #IAO rdms
  s=m.get_ovlp()
  M=m.mo_coeff
  M=reduce(np.dot,(a.T,s,M))
  iao_dm_list=np.einsum('ijkl,mk->ijml',dm_list,M)
  iao_dm_list=np.einsum('ijml,nl->ijmn',iao_dm_list,M)

  ##################
  #Energy
  e=np.einsum('ijll,l->ij',dm_list,m.mo_energy)
  e=e[:,0]+e[:,1]
  e-=e[0] #Eigenvalue difference

  #Number occupations 
  n=np.einsum('ijmm->ijm',iao_dm_list)
  n=n[:,0,:]+n[:,1,:]
  labels=['yz','y','xz','x','3dz2','pz','s']
  
  #Data object
  d=np.concatenate((e[:,np.newaxis],n,np.ones(N+1)[:,np.newaxis]*mol_spin),axis=1)
  if(data is None): data=d
  else: data=np.concatenate((data,d),axis=0)

#Full data frame
df=pd.DataFrame(data,columns=["E"]+list(labels)+["spin"])
df['E']*=27.2114

#Dump 
df.to_pickle(f.split(".")[0]+"_"+str(detgen)+"_N"+str(N)+"_Ndet"+str(Ndet)+"_c"+str(c)+"_beta"+str(beta)+".pickle")
