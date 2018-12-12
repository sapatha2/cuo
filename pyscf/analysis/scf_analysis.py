#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf, fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol

charge=0
#for mol_spin in [1,3]:
#for r in [1.725,1.963925]:
#for method in ['ROHF','B3LYP','PBE0']:
#for basis in ['vdz','vtz']:

for mol_spin in [3]:
  for r in [1.963925]:
    for method in ['B3LYP']:
      for basis in ['vtz']:
        for el in ['Cu']:
          for charge in [0]:
            chkfile="../chkfiles/"+el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method+".chk"
            mol=lib.chkfile.load_mol(chkfile)
            
            if("U" in method): m=UHF(mol)
            else: m=ROHF(mol)
            m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

            #Generate files for plotting orbitals in ../orbs
            print_qwalk_mol(mol,m,basename="../orbs/"+el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method)
            if(m.mo_energy.shape[0]==2):
              plt.plot(m.mo_energy[0],'*',label=method)
              plt.plot(m.mo_energy[1],'*')
            else:
              plt.plot(m.mo_energy,'o',label=method)
              
#Generate s(#)_eigenvalue_comp.pdf
plt.title("Spin="+str(mol_spin)+" eigenvalue comparison")
plt.legend(loc='best')
plt.show()
