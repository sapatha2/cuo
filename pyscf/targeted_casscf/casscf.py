#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from pyscf.mcscf import newton_casscf
from pyscf2qwalk import print_qwalk,print_cas_slater

#Run CASSCF vtz
chkfile="rohf/Cuvdz_r1.725_s1_B3LYP_0.chk"
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile,'scf'))

ncore=4
ncas=10
nelecas=(9,8)
ss = 0.75
symm = 'E2x'
for state_id in np.arange(10):
  casscf=mcscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
  casscf.frozen=[0,1,2,3]
  casscf.fcisolver.wfnsym=symm
  casscf.diis=scf.ADIIS
  casscf.fix_spin_(ss=ss)

  casscf.chkfile='vdz_'+casscf.fcisolver.wfnsym+'_ss'+str(ss)+'_casscf'+str(state_id)+'.chk'

  casscf.kernel()
  casscf.verbose=4
  casscf.analyze()
