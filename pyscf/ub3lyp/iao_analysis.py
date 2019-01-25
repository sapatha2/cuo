#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from functools import reduce 

#for mol_spin in [1,3]:
#for r in [1.725,1.963925]:
#for method in ['ROHF','B3LYP','PBE0']:
#for basis in ['vdz','vtz']:

#Gather IAOs
f='b3lyp_iao_b.pickle'
a=np.load(f)
print(a.shape)
           
charge=0
S=1
r=1.725
method='UB3LYP'
basis='vtz'
el='Cu'

for run in range(8):
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

  #plt.title("S="+str(mol_spin))
  #plt.matshow(dm_u+dm_d - np.diag(np.diag(dm_u+dm_d)),vmin=-1,vmax=1,cmap=plt.cm.bwr)
  #plt.show()

  #Check traces
  act=np.array([1,5,6,7,8,9,11,12,13])
  labels=["3s","4s","3px","3py","3pz","3dxy","3dyz","3dz2","3dxz","3dx2y2","2s","2px","2py","2pz"]
  print('Full trace: ', np.trace(dm_u),np.trace(dm_d))                         #Full Trace
  if(dm_u.shape[0]>12):
    print('Active trace: ',sum(np.diag(dm_u)[act]),sum(np.diag(dm_d)[act]))      #Active Trace
  
  #Check e matrix
  '''
  s=m.get_ovlp()
  H1=np.diag(m.mo_energy)
  e1=reduce(np.dot,(a.T,s,m.mo_coeff,H1,m.mo_coeff.T,s.T,a))*27.2114
  e1=(e1+e1.T)/2.
  labels=["3s","4s","3px","3py","3pz","3dxy","3dyz","3dz2","3dxz","3dx2y2","2s","2px","2py","2pz"]
  '''

  '''
  fig=plt.figure()
  ax=fig.add_subplot(111)
  cax=ax.matshow(e1,vmin=-1,vmax=1,cmap=plt.cm.bwr)
  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  ax.set_xticklabels(labels,rotation=90)
  ax.set_yticklabels(labels)
  plt.show()
  '''
  
  #Eigenvalue comparison
  '''
  w,__=np.linalg.eigh(e1)
  plt.plot(sorted(m.mo_energy[:len(w)]*27.2114),'go',label='MO')
  plt.plot(sorted(w),'b*',label='IAO')
  plt.xlabel('Eigenvalue')
  plt.ylabel('Energy (eV)')
  plt.title("DFT vs. IAO igenvalues")
  plt.savefig("evals_"+str(run)+".pdf",bbox_inches='tight')
  plt.close()
  '''

  '''
  #Plot IAOs
  for i in range(a.shape[1]):
    m.mo_coeff[:,i]=a[:,i] 
  print_qwalk_mol(mol,m,basename="../orbs/b3lyp_iao_b") 
  exit(0)
  '''
