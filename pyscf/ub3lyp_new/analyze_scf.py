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

chkfile="Cuvtz_r1.725_s1_UB3LYP_12.chk"
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile,'scf'))
print(sum(m.mo_occ[0]),sum(m.mo_occ[1]))
print_qwalk_mol(mol,m,method='scf',basename='Cuvtz_r1.725_s1_UB3LYP_12')
