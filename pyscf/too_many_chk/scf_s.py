#PySCF input file for CuO calculations 
import json
from pyscf import gto,scf,mcscf, fci,lo,ci,cc
from pyscf.scf import ROHF,ROKS,UHF,UKS, addons
import numpy as np
import pandas as pd

df=json.load(open("trail.json"))
core=[3,3,1,1,1,1,0,0,0,0]
base=[3,3,2,2,2,2,1,1,1,1]
def remove(obj,n):
  ret_obj=[]
  if(n==0): return [obj]
  for i in range(1,n+1):   
    for j in range(len(obj)):
      tmp=obj[:]
      tmp[j]-=i
      if(tmp[j]>=0): ret_obj+=remove(tmp,n-i)
  return ret_obj
r=remove(base,3)
r=list(set([tuple(x) for x in r]))
r=np.array(r)+np.array(core)
print(np.sum(r,axis=1))

Svec=[1,-1]*5
Lvec=[0,0]*1+[1,-1]*2+[2,-2]*2
S=np.dot(r,Svec)
L=np.dot(r,Lvec)
symm_dict=[{'A1':(x[0],x[1]),'E1x':(x[2],x[3]),'E1y':(x[4],x[5]),
'E2x':(x[6],x[7]),'E2y':(x[8],x[9])} for x in r]
charge=0

datacsv={}
for nm in['run','method','basis','pseudopotential','bond-length','S','L','E','conv']:
  datacsv[nm]=[]

for run in range(len(r)):
  for r in [1.725]:
    for method in ['UB3LYP']:
      for basis in ['vdz','vtz']:
        for el in ['Cu']:
          if(S[run]>0):          
            molname=el+'O'
            mol=gto.Mole()

            mol.ecp={}
            mol.basis={}
            for e in [el,'O']:
              mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
              mol.basis[e]=gto.basis.parse(df[e][basis])
            mol.charge=charge
            mol.spin=S[run]
            mol.build(atom="%s 0. 0. 0.; O 0. 0. %g"%(el,r),verbose=4,symmetry=True)
            
            if basis=='vdz':
              #These are the orbitals for which we want to read-in an initial DM guess 
              TM_3s_orbitals = []
              TM_4s_orbitals = []
              TM_3p_orbitals = []
              TM_3d_orbitals = []
              O_2s_orbitals  = []
              O_2p_orbitals  = []

              aos=mol.ao_labels()
              print('')
              print('AO labels')
              print(aos)
              print('')
              for i,x in enumerate(aos):

                #Find the TM 3s labels
                if (('3s' in x) and (el in x)):
                  TM_3s_orbitals.append(i)

                #Find the TM 4s labels
                if (('4s' in x) and (el in x)):
                  TM_4s_orbitals.append(i)

                #Find the TM 3p labels
                if (('3p' in x) and (el in x)):
                  TM_3p_orbitals.append(i)

                #Find the TM 3d labels
                if (('3d' in x) and (el in x)):
                  TM_3d_orbitals.append(i)

                #Find the O 2s labels
                if (('2s' in x) and ('O' in x)):
                  O_2s_orbitals.append(i)

                #Find the O 2p labels
                if (('2p' in x) and ('O' in x)):
                  O_2p_orbitals.append(i)

              #There should be 5 3d TM orbitals. Let's check this!
              assert len(TM_3d_orbitals)==5     

              ##############################################################################################
              if("U" in method): 
                if("HF" in method): 
                  m=UHF(mol)
                else:
                  m=UKS(mol)
                  m.xc=method[1:]
              else: 
                if(method=="ROHF"):
                  m=ROHF(mol)
                else:
                  m=ROKS(mol)
                  m.xc=method
              ##############################################################################################
              
              dm=np.zeros(m.init_guess_by_minao().shape)
              
              #The 3s is always doubly-occupied for the TM atom
              for s in TM_3s_orbitals:
                for spin in [0,1]:
                  dm[spin,s,s]=1
       
              #The 4s is always at least singly-occupied for the TM atom
              for s in TM_4s_orbitals:
                dm[0,s,s]=1
        
              #Control the 4s double-occupancy 
              if (el=='Cr'):
                for s in TM_4s_orbitals:
                  print('We are singly filling this 4s-orbital: '+np.str(aos[s]) )
                  dm[1,s,s]=0

              #Always doubly-occupy the 3p orbitals for the TM atom
              for p in TM_3p_orbitals:
                for s in [0,1]:
                  dm[s,p,p]=1

              #Control the 3d occupancy for CrO...
              if (el=='Cr'):
                for i,d in enumerate(TM_3d_orbitals):

                  #These are the 3d orbitals we want to fill to get the correct symmetry
                  if ( ('xy' in aos[d]) or ('yz' in aos[d]) or ('z^2' in aos[d]) or ('x2-y2' in aos[d]) ):
                    print('We are singly filling this d-orbital: '+np.str(aos[d]) )
                    dm[0,d,d]=1
      
              m.chkfile=el+basis+"_r"+str(r)+"_l"+str(L[run])+"_s"+str(S[run])+"_"+method+"_"+str(run)+".chk"
              m.irrep_nelec = symm_dict[run]
              m.max_cycle=100
              m = addons.remove_linear_dep_(m)
              m.conv_tol=1e-6
              
              #Only need an initial guess for CrO and CuO...
              if (el=='Cr' or el=='Cu'):
                total_energy=m.kernel(dm)
              else:
                total_energy=m.kernel()
         
              #Compute the Mulliken orbital occupancies...
              m.analyze()
              assert(np.sum(m.mo_occ)==25)

            #Once we get past the vdz basis, just read-in the existing chk file...
            else:
              ##############################################################################################
              if("U" in method): 
                if("HF" in method): 
                  m=UHF(mol)
                else:
                  m=UKS(mol)
                  m.xc=method[1:]
              else: 
                if(method=="ROHF"):
                  m=ROHF(mol)
                else:
                  m=ROKS(mol)
                  m.xc=method
              ##############################################################################################
              
              dm=m.from_chk(el+'vdz'+"_r"+str(r)+"_l"+str(L[run])+"_s"+str(S[run])+"_"+method+"_"+str(run)+".chk")
              m.chkfile=el+basis+"_r"+str(r)+"_l"+str(L[run])+"_s"+str(S[run])+"_"+method+"_"+str(run)+".chk"
              m.irrep_nelec = symm_dict[run]
              m.max_cycle=100
              m = addons.remove_linear_dep_(m)
              m.conv_tol=1e-6
              total_energy=m.kernel(dm)
              m.analyze()
              assert(np.sum(m.mo_occ)==25)

            datacsv['run'].append(run)
            datacsv['bond-length'].append(r)
            datacsv['L'].append(L[run])
            datacsv['S'].append(S[run])
            datacsv['method'].append(method)
            datacsv['basis'].append(basis)
            datacsv['pseudopotential'].append('trail')
            datacsv['E'].append(total_energy)
            datacsv['conv'].append(m.converged)
            pd.DataFrame(datacsv).to_csv("cuo.csv",index=False)
