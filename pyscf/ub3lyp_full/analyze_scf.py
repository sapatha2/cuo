#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from pyscf.mcscf import newton_casscf
from pyscf2qwalk import print_qwalk_mol


el='Cu'
r=1.725
method='UB3LYP'
S=3
for run in range(6):
  chkfile=el+'vtz'+"_r"+str(r)+"_s"+str(S)+"_"+method+"_"+str(run)+".chk"
  mol=lib.chkfile.load_mol(chkfile)
  m=ROHF(mol)
  m.__dict__.update(lib.chkfile.load(chkfile,'scf'))
  print(sum(m.mo_occ[0]),sum(m.mo_occ[1]))
  print_qwalk_mol(mol,m,method='scf',basename='qwalk_vtz/gs_hispin'+str(run))
exit(0)

df=pd.DataFrame.from_csv('cuo_u.csv')
df=df[df['basis']=='vtz']
df['E']-=min(df['E'])
df['E']*=27.2114
print(df)
