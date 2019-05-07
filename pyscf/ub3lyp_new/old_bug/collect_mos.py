#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol

chkfiles=['../ub3lyp_full/Cuvtz_r1.725_s1_UB3LYP_'+str(i)+'.chk' for i in range(11)]
chkfiles+=['Cuvtz_r1.725_s1_UB3LYP_11.chk','Cuvtz_r1.725_s1_UB3LYP_12.chk']
occ=np.arange(14) 
mo_coeff=None

for chkfile in chkfiles:
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
 
  if(mo_coeff is None): 
    mo_coeff=[None,None]
    mo_coeff[0]=m.mo_coeff[0][:,occ]
    mo_coeff[1]=m.mo_coeff[1][:,occ]
  else: 
    mo_coeff[0]=np.concatenate((mo_coeff[0],m.mo_coeff[0][:,occ]),axis=1)
    mo_coeff[1]=np.concatenate((mo_coeff[1],m.mo_coeff[1][:,occ]),axis=1)

#Write to file 
m.mo_coeff=mo_coeff
print(m.mo_coeff[0].shape)
print(m.mo_coeff[1].shape)
print_qwalk_mol(mol,m,basename="all_s1")

'''
chkfiles = ['../ub3lyp_full/Cuvtz_r1.725_s3_UB3LYP_'+str(i)+'.chk' for i in range(6)]
chkfiles += ['Cuvtz_r1.725_s3_UB3LYP_13.chk']
occ=np.arange(14) 
mo_coeff=None

for chkfile in chkfiles:
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  
  if(mo_coeff is None): 
    mo_coeff=[None,None]
    mo_coeff[0]=m.mo_coeff[0][:,occ]
    mo_coeff[1]=m.mo_coeff[1][:,occ]
  else: 
    mo_coeff[0]=np.concatenate((mo_coeff[0],m.mo_coeff[0][:,occ]),axis=1)
    mo_coeff[1]=np.concatenate((mo_coeff[1],m.mo_coeff[1][:,occ]),axis=1)

#Write to file 
m.mo_coeff=mo_coeff
print(m.mo_coeff[0].shape)
print(m.mo_coeff[1].shape)
print_qwalk_mol(mol,m,basename="all_s3")
'''
