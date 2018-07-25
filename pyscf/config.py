#Check the electronic configuration of the CuO ground state
import json
from pyscf import lib,gto,scf,mcscf, fci,lo,ci,cc
from pyscf.scf import ROHF, UHF,ROKS
import numpy as np
import pandas as pd

el='Cu'
charge=0
for basis in ['vdz','vtz']:
  chkfile=el+basis+str(charge)+".chk"
  mol=lib.chkfile.load_mol(chkfile)
  m = ROHF(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  print("####################################################\n basis="+basis)
  m.analyze()
  
