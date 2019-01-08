#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol

#for mol_spin in [1,3]:
#for r in [1.725,1.963925]:
#for method in ['ROHF','B3LYP','PBE0']:
#for basis in ['vdz','vtz']:

occ=np.arange(11,15)-1 #for basis with 2p, 4s only
#occ=np.arange(6,15)-1 #for bases without _full
#occ=np.arange(15)-1   #for bases with _full
mo_coeff=None
for mol_spin in [-1,1,3,5]:
  for r in [1.963925]:
    for method in ['B3LYP']:
      for basis in ['vtz']:
        for el in ['Cu']:
          for charge in [0]:
            chkfile="../chkfiles/"+el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method+".chk"
            if(mol_spin==5): chkfile="../full_chk/Cuvtz_r1.963925_c0_s1_B3LYP_2Y.chk"
            mol=lib.chkfile.load_mol(chkfile)
            
            if("U" in method): m=UHF(mol)
            else: m=ROHF(mol)
            m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

            #Collect IAOs for construction 
            if(mo_coeff is None): mo_coeff=m.mo_coeff[:,occ]
            else: mo_coeff=np.concatenate((mo_coeff,m.mo_coeff[:,occ]),axis=1)

#build minimum basis b3lyp_iao.pickle (cu: 1s, 1d; o: 1p)
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

#minimum basis using BFD PBC
#from sco_basis import minbasis

#Build IAOs
s=m.get_ovlp()
a=lo.iao.iao(mol, mo_coeff, minao=minbasis)
a=lo.vec_lowdin(a,s)
a.dump('b3lyp_iao_test.pickle')
