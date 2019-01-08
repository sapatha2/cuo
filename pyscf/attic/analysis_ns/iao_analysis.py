#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from functools import reduce 

#for mol_spin in [1,3]:
#for r in [1.725,1.963925]:
#for method in ['ROHF','B3LYP','PBE0']:
#for basis in ['vdz','vtz']:

#Gather IAOs
f='b3lyp_iao_b_full.pickle'
a=np.load(f)
print(a.shape)
mo_coeff=None
for mol_spin in [1,3]:
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

            #Build RDM on IAO basis 
            s=m.get_ovlp()
            mo_occ=np.array([np.ceil(m.mo_occ-m.mo_occ/2),np.floor(m.mo_occ/2)])
            M=m.mo_coeff[:,mo_occ[0]>0]
            M=reduce(np.dot,(a.T,s,M))
            dm_u=np.dot(M,M.T)
            M=m.mo_coeff[:,mo_occ[1]>0]
            M=reduce(np.dot,(a.T,s,M))
            dm_d=np.dot(M,M.T)
            
            #Check traces
            act=np.array([1,5,6,7,8,9,11,12,13])
            print('Full trace: ', np.trace(dm_u),np.trace(dm_d))                         #Full Trace
            print('Active trace: ',sum(np.diag(dm_u)[act]),sum(np.diag(dm_d)[act]))      #Active Trace
            '''
            #b3lyp_iao_b.pickle
            Full trace:  12.99494970270977 11.995149691084729
            Active trace:  7.999910633007202 7.000125945705045
            Full trace:  13.97435166516741 10.994493591931247
            Active trace:  8.97499800166205 6.000224099450173
            '''
            '''
            #b3lyp_iao_b_full.pickle
	    Full trace:  12.996569407483594 11.996769393107362
	    Active trace:  8.018621997591124 7.018884670023201
	    Full trace:  13.975010186039171 10.995150873644418
	    Active trace:  8.975771998268106 6.022182931843698 
            '''

            #Check e matrix
            s=m.get_ovlp()
            H1=np.diag(m.mo_energy)
            e1u=reduce(np.dot,(a.T,s,m.mo_coeff,H1,m.mo_coeff.T,s.T,a))
            e1u=(e1u+e1u.T)/2.
            labels=["3s","4s","3px","3py","3pz","3dxy","3dyz","3dz2","3dxz","3dx2y2","2s","2px","2py","2pz"]
            plt.matshow(e1u-np.diag(np.diag(e1u)),cmap=plt.cm.bwr)
            plt.xticks(np.arange(len(labels)),labels,rotation=90)
            plt.yticks(np.arange(len(labels)),labels)
            plt.colorbar()
            plt.show()
            exit(0)
            
            w,__=np.linalg.eigh(e1u)
            plt.plot(m.mo_energy[:len(w)],'go',label='MO')
            plt.plot(w,'b*',label='IAO')
            plt.xlabel('Eigenvalue')
            plt.ylabel('Energy (Ha)')
            plt.title(f.split(".")[0]+" S="+str(mol_spin)+" DFT eigenvalues")
            plt.savefig(f.split(".")[0]+"_s"+str(mol_spin)+"_evals.pdf",bbox_inches='tight')
            plt.close()

            #Plot orbitals
            '''
            for i in range(a.shape[1]):
              m.mo_coeff[:,i]=a[:,i] 
            print_qwalk_mol(mol,m,basename="../orbs/"+el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+f.split(".")[0]) 
            '''
