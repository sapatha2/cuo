#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from functools import reduce 
from downfold_tools import get_qwalk_dm, sum_onebody, sum_J, sum_U, sum_V

#for mol_spin in [1,3]:
#for r in [1.725,1.963925]:
#for method in ['ROHF','B3LYP','PBE0']:
#for basis in ['vdz','vtz']:

#Gather IAOs
f='b3lyp_iao_b_overshoot.pickle'
a=np.load(f)
#print(a.shape)
 
#Gather MOs
'''
chkfile='../chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
mol=lib.chkfile.load_mol(chkfile)
m=ROKS(mol)
m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
a=m.mo_coeff[:,:14]
print(a.shape)
'''

charge=0
S=1
r=1.725
method='UB3LYP'
basis='vtz'
el='Cu'
df=None
for run in range(16):
  chkfile=el+basis+"_r"+str(r)+"_s"+str(S)+"_"+method+"_"+str(run)+".chk"
  print(chkfile)
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

  #Build RDM on IAO basis 
  s=m.get_ovlp()
  mo_occ=m.mo_occ
  M=m.mo_coeff[0][:,mo_occ[0]>0]
  M=reduce(np.dot,(a.T,s,M))
  dm_u=np.dot(M,M.T)
  M=m.mo_coeff[1][:,mo_occ[1]>0]
  M=reduce(np.dot,(a.T,s,M))
  dm_d=np.dot(M,M.T)
  
  obdm=np.array([dm_u,dm_d])

  '''
  #Gather (MO)
  orb1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,5, 6 ,7 ,7 ,12]
  orb2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,10,11,12,13,13]
  one_body=sum_onebody(obdm,orb1,orb2)
  one_labels=['t_'+str(orb1[i])+'_'+str(orb2[i]) for i in range(len(orb1))]
  sigU=obdm[0][13,13]*obdm[1][13,13]
  '''

  '''
  #Gather (IAO)
  orb1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,1,1,6,7,8]
  orb2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,7,13,12,13,11]
  one_body=sum_onebody(obdm,orb1,orb2)
  one_labels=['t_'+str(orb1[i])+'_'+str(orb2[i]) for i in range(len(orb1))]
  
  sigU=obdm[0][1,1]*obdm[1][1,1]
  sigV=np.sum(sum_onebody(obdm,[5,6,7,8,9],[5,6,7,8,9]))*np.sum(sum_onebody(obdm,[11,12,13],[11,12,13]))

  dat=np.array([sigU,sigV]+list(one_body))
  d=pd.DataFrame(dat[:,np.newaxis].T,columns=['sigU','sigV']+one_labels)
  '''
 
  dat=np.diag(dm_u)+np.diag(dm_d)
  print(dat.shape)
  d=pd.DataFrame(dat[:,np.newaxis].T,columns=['n'+str(i) for i in range(len(dat))])
  d=d.astype('double')
  if(df is None): df=d
  else: df=pd.concat((df,d),axis=0)

fout='c.pickle'
print(df)
df.to_pickle(fout)
