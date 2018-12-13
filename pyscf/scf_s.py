#PySCF input file for CuO calculations 

import json
from pyscf import gto,scf,mcscf, fci,lo,ci,cc
from pyscf.scf import ROHF,ROKS,UHF,UKS, addons
import numpy as np
import pandas as pd

df=json.load(open("trail.json"))

#spins={'CuO':3}
#re={'CuO':1.725}
#re={'CuO':1.963925}
#nd={'Cu':(5,4)} 

symm_dict={}
#symm_dict['CuO0_1']={'A1' :(5,5), 'E1x': (3,3), 'E1y': (3,2), 'E2x': (1,1), 'E2y': (1,1)}
#symm_dict['CuO0_3']={'A1' :(6,5), 'E1x': (3,2), 'E1y': (3,2), 'E2x': (1,1), 'E2y': (1,1)}

#SECOND CALCULATION TO GET IAOS SYMMETRIC!
symm_dict['CuO0_1']={'A1' :(5,5), 'E1x': (3,2), 'E1y': (3,3), 'E2x': (1,1), 'E2y': (1,1)}

datacsv={}
for nm in ['molecule','bond-length','charge','spin','method','basis','pseudopotential','totalenergy',
           'totalenergy-stocherr','totalenergy-syserr','pyscf-version']:
  datacsv[nm]=[]

for mol_spin in [1]:
  #for r in [1.725,1.963925]:
  for r in [1.963925]:
    #for method in ['UHF','UB3LYP','UPBE0','ROHF','B3LYP','PBE0']:
    for method in ['B3LYP']:
      for basis in ['vdz','vtz']:
        for el in ['Cu']:
          for charge in [0]: 
            molname=el+'O'
            mol=gto.Mole()

            mol.ecp={}
            mol.basis={}
            for e in [el,'O']:
              mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
              mol.basis[e]=gto.basis.parse(df[e][basis])
            mol.charge=charge
            mol.spin=mol_spin
            #mol.build(atom="%s 0. 0. 0.; O 0. 0. %g"%(el,re[molname]),verbose=4)
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
      
              m.chkfile=el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method+".chk"
              m.irrep_nelec = symm_dict[el+'O'+str(charge)+'_'+str(mol.spin)]
              m.max_cycle=100
              m = addons.remove_linear_dep_(m)
              #m.direct_scf_tol=1e-5
              m.conv_tol=1e-6
              
              #m.level_shift_factor=0.1
              #m=scf.newton(m)
              
              #Only need an initial guess for CrO and CuO...
              if (el=='Cr' or el=='Cu'):
                total_energy=m.kernel(dm)
              else:
                total_energy=m.kernel()
         
              #Compute the Mulliken orbital occupancies...
              m.analyze()

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
              
              #dm=m.from_chk(el+'vdz'+str(charge)+"_"+method+".chk")
              #m.chkfile=el+basis+str(charge)+"_"+method+".chk"        
              dm=m.from_chk(el+'vdz'+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method+".chk")
              m.chkfile=el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method+".chk"
              #m=scf.newton(m)
              m.irrep_nelec = symm_dict[el+'O'+str(charge)+'_'+str(mol.spin)]
              m.max_cycle=100
              m = addons.remove_linear_dep_(m)
              #m.direct_scf_tol=1e-5
              m.conv_tol=1e-6
              total_energy=m.kernel(dm)
              m.analyze()

            '''
            datacsv['molecule'].append(molname)
            datacsv['bond-length'].append(r)
            datacsv['charge'].append(charge)
            datacsv['spin'].append(mol_spin)
            datacsv['method'].append(method)
            datacsv['basis'].append(basis)
            datacsv['pseudopotential'].append('trail')
            datacsv['totalenergy'].append(total_energy)
            datacsv['totalenergy-stocherr'].append(0.0)
            datacsv['totalenergy-syserr'].append(0.0)
            datacsv['pyscf-version'].append('new')
            pd.DataFrame(datacsv).to_csv("cuo_full_s.csv",index=False)
            '''
