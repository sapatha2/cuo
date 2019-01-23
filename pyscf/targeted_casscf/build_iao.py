#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol

chkfiles=['sig/vtz_spin1_casscf0','pi/vtz_spin1_casscf0','pi/vtz_spin1_casscf1','pi/vtz_spin1_casscf2',
'sig/vtz_spin3_casscf0','pi/vtz_spin3_casscf0']
chkfiles+=[x+'mirror' for x in chkfiles]

occ=np.arange(5,15)-1 #up to 4s orbitals
mo_coeff=None
for f in chkfiles:
  mo=lib.chkfile.load(f+'.chk','mcscf/mo_coeff')
  if(mo_coeff is None): mo_coeff=mo[:,occ]
  else: mo_coeff=np.concatenate((mo_coeff,mo[:,occ]),axis=1)

#build minimum basis b3lyp_iao_b.pickle (cu: 2s, 1p, 1d; o: 1s, 1p)
cu_basis=[]

mol=lib.chkfile.load_mol(f+'.chk')
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load('rohf/Cuvtz_r1.725_s1_ROHF_0.chk', 'scf'))

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

#Plot IAOs
m.mo_coeff[:,:a.shape[1]]=a
print_qwalk_mol(mol,m,method='scf',basename='qwalk/b3lyp_iao_b')
