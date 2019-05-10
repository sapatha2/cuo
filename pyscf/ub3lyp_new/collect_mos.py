#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
'''
chkfiles=['../ub3lyp_full/Cuvtz_r1.725_s1_UB3LYP_'+str(i)+'.chk' for i in range(11)]
chkfiles+=['Cuvtz_r1.725_s1_UB3LYP_11.chk','Cuvtz_r1.725_s1_UB3LYP_12.chk']
mo_coeff=None
for chkfile in chkfiles:
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  print(chkfile)
  print(m.mo_occ[0])
  print(m.mo_occ[1])

  occ0 = m.mo_occ[0][:14]
  occ1 = m.mo_occ[1][:14]
  
  occ0 = np.argsort(-occ0)
  occ1 = np.argsort(-occ1)
  print(occ0)
  print(occ1)
  
  if(mo_coeff is None): 
    mo_coeff=[None,None]
    mo_coeff[0]=m.mo_coeff[0][:,occ0]
    mo_coeff[1]=m.mo_coeff[1][:,occ1]
  else: 
    mo_coeff[0]=np.concatenate((mo_coeff[0],m.mo_coeff[0][:,occ0]),axis=1)
    mo_coeff[1]=np.concatenate((mo_coeff[1],m.mo_coeff[1][:,occ1]),axis=1)
#Write to file 
m.mo_coeff=mo_coeff
print(m.mo_coeff[0].shape)
print(m.mo_coeff[1].shape)
print_qwalk_mol(mol,m,basename="all_s1")
'''

'''
chkfiles = ['../ub3lyp_full/Cuvtz_r1.725_s3_UB3LYP_'+str(i)+'.chk' for i in range(6)]
chkfiles += ['Cuvtz_r1.725_s3_UB3LYP_13.chk']
mo_coeff=None
for chkfile in chkfiles:
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  print(chkfile)
  print(m.mo_occ[0])
  print(m.mo_occ[1])

  occ0 = m.mo_occ[0][:14]
  occ1 = m.mo_occ[1][:14]
  
  occ0 = np.argsort(-occ0)
  occ1 = np.argsort(-occ1)
  print(occ0)
  print(occ1)
  
  if(mo_coeff is None): 
    mo_coeff=[None,None]
    mo_coeff[0]=m.mo_coeff[0][:,occ0]
    mo_coeff[1]=m.mo_coeff[1][:,occ1]
  else: 
    mo_coeff[0]=np.concatenate((mo_coeff[0],m.mo_coeff[0][:,occ0]),axis=1)
    mo_coeff[1]=np.concatenate((mo_coeff[1],m.mo_coeff[1][:,occ1]),axis=1)
#Write to file 
m.mo_coeff=mo_coeff
print(m.mo_coeff[0].shape)
print(m.mo_coeff[1].shape)
print_qwalk_mol(mol,m,basename="all_s3")
'''

'''
chkfiles=['../ub3lyp_full/Cuvtz_r1.725_s1_UB3LYP_'+str(i)+'.chk' for i in [0,2,3,4]]
mo_coeff=None
for chkfile in chkfiles:
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  print(chkfile)
  print(m.mo_occ[0])
  print(m.mo_occ[1])

  occ0 = m.mo_occ[0][:14]
  occ1 = m.mo_occ[1][:14]
  
  occ0 = np.argsort(-occ0)
  occ1 = np.argsort(-occ1)
  print(occ0)
  print(occ1)
  
  if(mo_coeff is None): 
    mo_coeff=[None,None]
    mo_coeff[0]=m.mo_coeff[0][:,occ0]
    mo_coeff[1]=m.mo_coeff[1][:,occ1]
  else: 
    mo_coeff[0]=np.concatenate((mo_coeff[0],m.mo_coeff[0][:,occ0]),axis=1)
    mo_coeff[1]=np.concatenate((mo_coeff[1],m.mo_coeff[1][:,occ1]),axis=1)
#Write to file 
m.mo_coeff=mo_coeff
print(m.mo_coeff[0].shape)
print(m.mo_coeff[1].shape)
print_qwalk_mol(mol,m,basename="all_1extra")
'''

chkfiles = ['../ub3lyp_full/Cuvtz_r1.725_s3_UB3LYP_'+str(i)+'.chk' for i in range(2)]
mo_coeff=None
for chkfile in chkfiles:
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  print(chkfile)
  print(m.mo_occ[0])
  print(m.mo_occ[1])

  occ0 = m.mo_occ[0][:14]
  occ1 = m.mo_occ[1][:14]
  
  occ0 = np.argsort(-occ0)
  occ1 = np.argsort(-occ1)
  print(occ0)
  print(occ1)
  
  if(mo_coeff is None): 
    mo_coeff=[None,None]
    mo_coeff[0]=m.mo_coeff[0][:,occ0]
    mo_coeff[1]=m.mo_coeff[1][:,occ1]
  else: 
    mo_coeff[0]=np.concatenate((mo_coeff[0],m.mo_coeff[0][:,occ0]),axis=1)
    mo_coeff[1]=np.concatenate((mo_coeff[1],m.mo_coeff[1][:,occ1]),axis=1)
#Write to file 
m.mo_coeff=mo_coeff
print(m.mo_coeff[0].shape)
print(m.mo_coeff[1].shape)
print_qwalk_mol(mol,m,basename="all_3extra")
