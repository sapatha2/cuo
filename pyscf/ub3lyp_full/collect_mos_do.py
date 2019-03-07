#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol

charge=0
S=1
r=1.725
method='UB3LYP'
basis='vtz'
el='Cu'

occ=np.arange(14) 
mo_coeff=None
for run in range(11):
  chkfile=el+basis+"_r"+str(r)+"_s"+str(S)+"_"+method+"_"+str(run)+".chk"
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
print_qwalk_mol(mol,m,basename="all_1do")
