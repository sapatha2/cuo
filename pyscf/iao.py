#Generate IAOs for the particular ground states
import json
from pyscf import lib,gto,scf,mcscf, fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from functools import reduce

el='Cu'
charge=0
minao={}
  
df=json.load(open("minao.json"))
for e in ['Cu','O']:
  minao[e]=gto.basis.parse(df[e])

for method in ['ROHF','B3LYP','PBE0']:
  for basis in ['vdz','vtz']:
    chkfile=el+basis+str(charge)+"_"+method+".chk"
    mol=lib.chkfile.load_mol(chkfile)
    m = ROHF(mol)
    m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

    print("####################################################\n method="+\
    method+", basis="+basis+"\n####################################################")
    homo=len(m.mo_occ[m.mo_occ>0])-1

    ##############################
    #MINAO VARIABLE HAS TO BE APPROPRIATELY DEFINED
    s=m.get_ovlp()
    mo_occ=m.mo_coeff[:,:homo+1]
    a=lo.iao.iao(mol, mo_occ, minao=minao)
    a=lo.vec_lowdin(a,s)
    mo_occ=reduce(np.dot,(a.T,s,mo_occ))
    dm_old=np.identity(homo+1)
    dm_u=np.dot(mo_occ,np.dot(dm_old,mo_occ.T))

    dm_old[homo,homo]=0
    dm_d=np.dot(mo_occ,np.dot(dm_old,mo_occ.T))

    #Save to qwalk orb files
    for i in range(a.shape[1]):
      m.mo_coeff[:,i]=a[:,i]
    print_qwalk_mol(mol,m,basename="../qwalk/"+el+basis+str(charge)+"_"+method+"iao")

    #Print traces and plot occ numbers
    print(np.trace(dm_d)+np.trace(dm_u))
    '''
    plt.subplot(211)
    if(basis=='vdz'): 
      plt.plot(np.diag(dm_u),'ko')
    else:
      plt.plot(np.diag(dm_u),'ks')
    plt.subplot(212)
    if(basis=='vtz'): 
      plt.plot(np.diag(dm_d),'ro')
    else:
      plt.plot(np.diag(dm_d),'rs')
    '''
  #plt.show()
