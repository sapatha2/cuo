#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf, fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol

el='Cu'
charge=0
i=0
for basis in ['vdz','vtz']:
  for method in ['ROHF','B3LYP','PBE0']:
    chkfile=el+basis+str(charge)+"_"+method+".chk"
    mol=lib.chkfile.load_mol(chkfile)
    m = ROHF(mol)
    m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
   
    print("####################################################\n method="+\
    method+", basis="+basis+"\n####################################################")
    e=np.array(m.mo_energy)
    homo=len(e[m.mo_occ>0])-1

    #Check the electronic configuration of the CuO ground state
    #m.analyze()
   
    #Check the type of orbital excitations
    '''
    print(m.mo_energy[homo-2])
    print(m.mo_coeff[:,homo-2][92])
    print(m.mo_coeff[:,homo-2][93])
    print(m.mo_coeff[:,homo-2][94])
    print(m.mo_energy[homo-1])
    print(m.mo_coeff[:,homo-1][92])
    print(m.mo_coeff[:,homo-1][93])
    print(m.mo_coeff[:,homo-1][94])
    print(m.mo_energy[homo])
    print(m.mo_coeff[:,homo][92])
    print(m.mo_coeff[:,homo][93])
    print(m.mo_coeff[:,homo][94])
    '''

    #Check the eigenvalues 
    e-=e[homo]
    e*=27.2
    
    plt.subplot(231+i) 
    plt.title(method+" "+basis)
    plt.ylabel("SCF eigenvalue-HOMO, eV")
    plt.xlabel("orbital number")
    plt.plot(np.arange(homo-10,homo+10),e[homo-10:homo+10],'bo')
    plt.plot(np.arange(homo-10,homo+1),e[m.mo_occ>0][homo-10:],'go')
    plt.plot(homo,e[homo],'rs')
    i+=1

    #print_qwalk_mol(mol,m,basename="../qwalk/"+el+basis+str(charge)+"_"+method)
  
plt.show()

