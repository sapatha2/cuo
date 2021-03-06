#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol

charge=0
S=[1,1,3,3,1,3]
r=1.725
method='B3LYP'
basis='vtz'
el='Cu'

occ=np.arange(14) 
mo_coeff=None
for run in range(len(S)):
  chkfile="../chk/"+el+basis+"_r"+str(r)+"_s"+str(S[run])+"_"+method+"_"+str(run)+".chk"
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  
  if(mo_coeff is None): mo_coeff=m.mo_coeff[:,occ]
  else: mo_coeff=np.concatenate((mo_coeff,m.mo_coeff[:,occ]),axis=1)

#Write to file 
m.mo_coeff=mo_coeff
print_qwalk_mol(mol,m,basename="../orbs/all")
