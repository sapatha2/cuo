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
<<<<<<< HEAD
S=1
for run in range(10,11):
=======
S=3
for run in range(6):
>>>>>>> ba5cb5d2c420434d93aeb8fb854b6ac770efe311
  chkfile=el+'vtz'+"_r"+str(r)+"_s"+str(S)+"_"+method+"_"+str(run)+".chk"
  mol=lib.chkfile.load_mol(chkfile)
  m=ROHF(mol)
  m.__dict__.update(lib.chkfile.load(chkfile,'scf'))
  print(sum(m.mo_occ[0]),sum(m.mo_occ[1]))
<<<<<<< HEAD
  print_qwalk_mol(mol,m,method='scf',basename='qwalk_vtz/gs_do'+str(run))
=======
  print_qwalk_mol(mol,m,method='scf',basename='qwalk_vtz/gs3_'+str(run))
>>>>>>> ba5cb5d2c420434d93aeb8fb854b6ac770efe311
exit(0)

#df=pd.DataFrame.from_csv('cuo_u.csv')
#df=df[df['basis']=='vtz']
#df['E']-=min(df['E'])
#df['E']*=27.2114
#print(df)
