#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS,UKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from functools import reduce 

#MO basis 
'''
chkfile='../chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
mol=lib.chkfile.load_mol(chkfile)
m=ROKS(mol)
m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
a=m.mo_coeff[:,:14]
'''
a = pd.read_pickle('../ub3lyp_full/b3lyp_iao_b.pickle')

charge=0
S=1
r=1.725
method='UB3LYP'
basis='vtz'
el='Cu'
 
for chkfile in ['Cuvtz_r1.725_s1_UB3LYP_11.chk',
'Cuvtz_r1.725_s1_UB3LYP_12.chk','Cuvtz_r1.725_s1_UB3LYP_13.chk',
'Cuvtz_r1.725_s1_UB3LYP_14.chk']:
  mol=lib.chkfile.load_mol(chkfile)
  m=UKS(mol)
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


  #MOs
  #labels=['dx','dy','dz','dd','dd','px','py','pz','4s']
  #dm_u = dm_u[5:][:,5:]
  #dm_d = dm_d[5:][:,5:]

  #IAOs
  labels=['4s','dd','dy','dz','dx','dd','px','py','pz']
  ind=[1,5,6,7,8,9,11,12,13]
  dm_u = dm_u[ind][:,ind]
  dm_d = dm_d[ind][:,ind]

  #IAOs
  plt.matshow(dm_u,vmin=-1,vmax=1,cmap=plt.cm.bwr)
  plt.xticks(np.arange(len(labels)),labels)
  plt.yticks(np.arange(len(labels)),labels)
  plt.show()

  plt.matshow(dm_d,vmin=-1,vmax=1,cmap=plt.cm.bwr)
  plt.xticks(np.arange(len(labels)),labels)
  plt.yticks(np.arange(len(labels)),labels)
  plt.show()
