#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol

charge=0
S=[1,1,3,3,1,3,1,3,1,3]
r=1.725
method='B3LYP'
basis='vtz'
el='Cu'

occ=np.arange(6,15)-1 
mo_coeff=None
for run in range(len(S)):
  if(run>5): chkfile="../chk/"+el+basis+"_r"+str(r)+"_s"+str(S[run])+"_"+method+"_"+str(run-6)+"mirror.chk"
  else: chkfile="../chk/"+el+basis+"_r"+str(r)+"_s"+str(S[run])+"_"+method+"_"+str(run)+".chk"
  print(chkfile)
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  
  if(mo_coeff is None): mo_coeff=m.mo_coeff[:,occ]
  else: mo_coeff=np.concatenate((mo_coeff,m.mo_coeff[:,occ]),axis=1)

#build minimum basis b3lyp_iao_b.pickle (cu: 2s, 1p, 1d; o: 1s, 1p)
cu_basis=[]
for i in (mol.basis["Cu"]):
  if(len(cu_basis)<2): 
    if(i[0]==0): cu_basis.append(i)
  elif(len(cu_basis)==2):
    if(i[0]==1): cu_basis.append(i)
  elif(len(cu_basis)==3):
    if(i[0]==2): cu_basis.append(i)
  else:
    pass
o_basis=[]
for i in (mol.basis["O"]):
  if(len(o_basis)==0): 
    if(i[0]==0): o_basis.append(i)
  elif(len(o_basis)==1): 
    if(i[0]==1): o_basis.append(i)
  else:
    pass
minbasis={'Cu':cu_basis,'O':o_basis}

#Build IAOs
s=m.get_ovlp()
a=lo.iao.iao(mol, mo_coeff, minao=minbasis)
a=lo.vec_lowdin(a,s)
a.dump('b3lyp_iao_b.pickle')
