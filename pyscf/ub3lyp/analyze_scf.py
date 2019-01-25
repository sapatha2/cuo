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

el='Cu'
r=1.725
method='UB3LYP'
S=1
for run in range(5):
  chkfile=el+'vdz'+"_r"+str(r)+"_s"+str(S)+"_"+method+"_"+str(run)+".chk"
  mol=lib.chkfile.load_mol(chkfile)
  m=ROHF(mol)
  m.__dict__.update(lib.chkfile.load(chkfile,'scf'))
  print_qwalk_mol(mol,m,method='scf',basename='qwalk/gs'+str(run))
