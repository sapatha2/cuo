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
f='b3lyp_iao_b.pickle'
a=np.load(f)
print(a.shape)
for mol_spin in [-1,1,3]:
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
            print(mo_occ)
            M=m.mo_coeff[:,mo_occ[0]>0]
            M=reduce(np.dot,(a.T,s,M))
            dm_u=np.dot(M,M.T)
            M=m.mo_coeff[:,mo_occ[1]>0]
            M=reduce(np.dot,(a.T,s,M))
            dm_d=np.dot(M,M.T)
            
            #Check traces
            act=np.array([1,5,6,7,8,9,11,12,13])
            print('Full trace: ', np.trace(dm_u),np.trace(dm_d))                         #Full Trace
            if(dm_u.shape[0]>12):
              print('Active trace: ',sum(np.diag(dm_u)[act]),sum(np.diag(dm_d)[act]))      #Active Trace
            print(np.diag(dm_u)[act],np.diag(dm_d)[act])
            
            ''' 
            #b3lyp_iao 
            Full trace:  8.840832814616048 7.841332423958144
            Full trace:  8.840832814609485 7.841332423951078
            Full trace:  8.981529038120833 6.940653762730744
            
            #b3lyp_iao_full
            Full trace:  8.841268343773583 7.84196451387073
	    Full trace:  8.841268343767048 7.841964513863693
	    Full trace:  8.981573438252646 6.941070240064234 
            
            #b3lyp_iao_b 
	    Full trace:  12.994552212911534 11.995048752791913
	    Active trace:  7.997022606928624 6.997526898722619
	    Full trace:  12.9945522129124 11.995048752792275
	    Active trace:  7.99702260692954 6.997526898723032
	    Full trace:  13.966681984807854 10.992999814606678
	    Active trace:  8.967423623715172 5.995491490895137
            
            #b3lyp_iao_b_full
	    Full trace:  12.996185405502084 11.996681924440646
	    Active trace:  8.018456851447063 7.019016832528233
	    Full trace:  12.996185405503008 11.996681924441067
	    Active trace:  8.018456851447853 7.019016832528523
	    Full trace:  13.96730922290277 10.99363331140566
	    Active trace:  8.968252085875948 6.020952811406707
            '''

            #Check e matrix
            s=m.get_ovlp()
            H1=np.diag(m.mo_energy)
            e1=reduce(np.dot,(a.T,s,m.mo_coeff,H1,m.mo_coeff.T,s.T,a))*27.2114
            e1=(e1+e1.T)/2.
            labels=["3s","4s","3px","3py","3pz","3dxy","3dyz","3dz2","3dxz","3dx2y2","2s","2px","2py","2pz"]
           
            '''
            #Rotated H1 matrix
            plt.title(f.split(".")[0]+" S="+str(mol_spin)+" H1")
            plt.matshow(e1-np.diag(np.diag(e1)),cmap=plt.cm.bwr,vmax=-1.7,vmin=1.7)
            plt.xticks(np.arange(len(labels)),labels,rotation=90)
            plt.yticks(np.arange(len(labels)),labels)
            plt.colorbar()
            plt.savefig(f.split(".")[0]+"_s"+str(mol_spin)+"_h1.pdf",bbox_inches='tight')
            '''
            
            #Eigenvalue comparison
            '''
            w,__=np.linalg.eigh(e1)
            plt.plot(m.mo_energy[:len(w)]*27.2114,'go',label='MO')
            plt.plot(w,'b*',label='IAO')
            plt.xlabel('Eigenvalue')
            plt.ylabel('Energy (eV)')
            plt.title(f.split(".")[0]+" S="+str(mol_spin)+" DFT eigenvalues")
            plt.savefig(f.split(".")[0]+"_s"+str(mol_spin)+"_evals.pdf",bbox_inches='tight')
            plt.close()
            '''

            '''
            #Plot orbitals
            for i in range(a.shape[1]):
              m.mo_coeff[:,i]=a[:,i] 
            print_qwalk_mol(mol,m,basename="../orbs/"+el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+f.split(".")[0]) 
            exit(0)
            '''
