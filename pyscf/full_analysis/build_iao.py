#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol

mo_coeff=None
occ=np.arange(6,15)-1 #for bases without _full
name=['2X','2Y','4SigmaM']
mol_spin={'2X':1,'2Y':1,'4SigmaM':3}
r=1.963925
method='B3LYP'
basis='vtz'
el='Cu'
charge=0

#Collect relevant MOs
for chkfile in ['../chkfiles/Cuvtz_r1.963925_c0_s-1_B3LYP.chk',
                '../chkfiles/Cuvtz_r1.963925_c0_s1_B3LYP.chk',              #2X
                '../full_chk/Cuvtz_r1.963925_c0_s1_B3LYP_2Y.chk',           #2Y
                '../chkfiles/Cuvtz_r1.963925_c0_s3_B3LYP.chk']:             #4SigmaM
  mol=lib.chkfile.load_mol(chkfile)
  m=ROHF(mol)
  m.__dict__.update(lib.chkfile.load(chkfile,'scf'))
  if(mo_coeff is None): mo_coeff=m.mo_coeff[:,occ]
  else: mo_coeff=np.concatenate((mo_coeff,m.mo_coeff[:,occ]),axis=1)


#build minimum basis b3lyp_iao_full.pickle (cu: 1s, 1d; o: 1p)
'''
cu_basis=[]
for i in (mol.basis["Cu"]):
  if(len(cu_basis)==0): 
    if(i[0]==0): cu_basis.append(i)
  elif(len(cu_basis)==1):
    if(i[0]==2): cu_basis.append(i)
  else:
    pass
o_basis=[]
for i in (mol.basis["O"]):
  if(len(o_basis)==0): 
    if(i[0]==1): o_basis.append(i)
  else:
    pass
minbasis={'cu':cu_basis,'o':o_basis}
'''

#build minimum basis b3lyp_iao_b_full.pickle (cu: 2s, 1p, 1d; o: 1s, 1p)
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

#minimum basis using BFD PBC
#from sco_basis import minbasis

#Build IAOs
s=m.get_ovlp()
a=lo.iao.iao(mol, mo_coeff, minao=minbasis)
a=lo.vec_lowdin(a,s)
a.dump('b3lyp_iao_b_full.pickle')
