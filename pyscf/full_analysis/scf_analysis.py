#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf, fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol

charge=0
for name in ['4Phi','4Delta','2Delta','4SigmaM']:
  mol_spin=int(name[0])-1
  for r in [1.963925]:
    for method in ['B3LYP']:
      for basis in ['vtz']:
        for el in ['Cu']:
          for charge in [0]:
            chkfile="../full_chk/"+el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method+"_"+name+".chk"
            mol=lib.chkfile.load_mol(chkfile)
            
            if("U" in method): m=UHF(mol)
            else: m=ROHF(mol)
            m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

            #Generate files for plotting orbitals in ../orbs
            #print_qwalk_mol(mol,m,basename="../full_orbs/"+el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method+"_"+name)
