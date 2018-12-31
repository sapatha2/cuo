#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from functools import reduce 

#Gather IAOs
f='b3lyp_iao_b_full.pickle'
a=np.load(f)
print(a.shape)

occ=np.arange(6,15)-1 #for bases without _full
name=['2X','2Y','4SigmaM']
mol_spin={'2X':1,'2Y':1,'4SigmaM':3}
r=1.963925
method='B3LYP'
basis='vtz'
el='Cu'
charge=0

#Gather states
for chkfile in ['../chkfiles/Cuvtz_r1.963925_c0_s-1_B3LYP.chk',
                '../chkfiles/Cuvtz_r1.963925_c0_s1_B3LYP.chk',              #2X 
                '../full_chk/Cuvtz_r1.963925_c0_s1_B3LYP_2Y.chk',           #2Y 
                '../chkfiles/Cuvtz_r1.963925_c0_s3_B3LYP.chk']:             #4SigmaM

  mol=lib.chkfile.load_mol(chkfile)
  m=ROHF(mol)
  m.__dict__.update(lib.chkfile.load(chkfile,'scf'))
  
  #Build RDM on IAO basis 
  s=m.get_ovlp()
  mo_occ=np.array([np.ceil(m.mo_occ-m.mo_occ/2),np.floor(m.mo_occ/2)])
  M=m.mo_coeff[:,mo_occ[0]>0]
  M=reduce(np.dot,(a.T,s,M))
  dm_u=np.dot(M,M.T)
  M=m.mo_coeff[:,mo_occ[1]>0]
  M=reduce(np.dot,(a.T,s,M))
  dm_d=np.dot(M,M.T)

  #Check traces
  act=np.array([1,5,6,7,8,9,11,12,13])
  labels=["3s","4s","3px","3py","3pz","3dxy","3dyz","3dz2","3dxz","3dx2y2","2s","2px","2py","2pz"]
  print('Full trace: ', np.trace(dm_u),np.trace(dm_d))                         #Full Trace
  if(dm_u.shape[0]>12):
    print('Active trace: ',sum(np.diag(dm_u)[act]),sum(np.diag(dm_d)[act]))      #Active Trace
  print(np.diag(dm_u)[act]+np.diag(dm_d)[act])
  print(labels)

  #Check e matrix
  s=m.get_ovlp()
  H1=np.diag(m.mo_energy)
  e1=reduce(np.dot,(a.T,s,m.mo_coeff,H1,m.mo_coeff.T,s.T,a))*27.2114
  e1=(e1+e1.T)/2.
  labels=["3s","4s","3px","3py","3pz","3dxy","3dyz","3dz2","3dxz","3dx2y2","2s","2px","2py","2pz"]
  
  #Eigenvalue comparison
  w,__=np.linalg.eigh(e1)
  plt.plot(m.mo_energy[:len(w)]*27.2114,'go',label='MO')
  plt.plot(w,'b*',label='IAO')
  plt.xlabel('Eigenvalue')
  plt.ylabel('Energy (eV)')
  plt.show()
  plt.close()

#Plot iaos
for i in range(a.shape[1]):
  m.mo_coeff[:,i]=a[:,i]
print_qwalk_mol(mol,m,basename="../full_orbs/b3lyp_iao_b_full")
