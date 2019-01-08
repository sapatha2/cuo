#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from functools import reduce 

#Gather optimized MOs 
f='b3lyp_mo_symm.pickle'
a=np.load(f)
flist=['../chkfiles/Cuvtz_r1.963925_c0_s-1_B3LYP.chk',
  '../chkfiles/Cuvtz_r1.963925_c0_s1_B3LYP.chk',
  '../full_chk/Cuvtz_r1.963925_c0_s1_B3LYP_2Y.chk',
  '../chkfiles/Cuvtz_r1.963925_c0_s3_B3LYP.chk']

for chkfile in flist:
  mol=lib.chkfile.load_mol(chkfile)
  m=ROHF(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

  #Build MO RDMs
  s=m.get_ovlp()
  rho1=np.diag(m.mo_occ)
  M=reduce(np.dot,(a.T,s,m.mo_coeff))
  dm=reduce(np.dot,(M,rho1,M.T))
  print(dm.shape,np.trace(dm))
