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

chkfile="rohf/Cuvtz_r1.725_s1_ROHF_0.chk"
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile,'scf'))

#Write qwalk input files 
'''
spin=[1,1,1,1,1,3,3]
symm=['A1','E1y','E1y','E1y','E2y','A1','E1y']
state_id=[0,0,1,2,5,0,0]
ncore=4
ncas=10
nelecas=(9,8)
for run in range(len(spin)):
  sym_dir='sig'
  if(symm[run]=='E1y'): sym_dir='pi'
  elif(symm[run]=='E2y'): sym_dir='del'
  casscf=mcscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id[run])
  casscf.frozen=[0,1,2,3] #3px, 3py, 3pz
  casscf.fix_spin_(ss=spin[run]/2.*(spin[run]/2.+1))
  casscf.fcisolver.wfnsym=symm[run]
  chkfile=sym_dir+'/vtz_spin'+str(spin[run])+'_casscf'+str(state_id[run])+'.chk'
  casscf.__dict__.update(lib.chkfile.load(chkfile,'mcscf'))
  casscf.kernel()

  #Write Qwalk files!
  print_qwalk_mol(mol,casscf,method='mcscf',tol=1e-3,basename='qwalk/'+sym_dir+str(state_id[run])+'S'+str(spin[run]))
'''
