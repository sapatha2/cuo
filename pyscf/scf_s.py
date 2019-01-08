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
Svec=[1,-1]*5
Lvec=[0,0]*1+[1,-1]*2+[2,-2]*2
S=np.dot(r,Svec)
L=np.dot(r,Lvec)
symm_dict=[{'A1':(x[0],x[1]),'E1x':(x[2],x[3]),'E1y':(x[4],x[5]),
'E2x':(x[6],x[7]),'E2y':(x[8],x[9])} for x in r]
method='B3LYP'

z=0
for run in [23]:
  #for run in range(len(r)):
  for r in [1.725,1.963925]:
      for basis in ['vdz','vtz']:
        for el in ['Cu']:
          molname=el+'O'
          mol=gto.Mole()
          
          if(S[run]<0): pass
          else:
            mol.ecp={}
            mol.basis={}
            for e in [el,'O']:
              mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
              mol.basis[e]=gto.basis.parse(df[e][basis])
            mol.charge=0
            mol.spin=S[run]
            mol.build(atom="%s 0. 0. 0.; O 0. 0. %g"%(el,r),verbose=4,symmetry=True)

            m=ROHF(mol)
            #m.chkfile=el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol.spin)+"_"+method+"_"+run+".chk"
            m.irrep_nelec = symm_dict[run]
            
            m.max_cycle=100
            m = addons.remove_linear_dep_(m)
            m.conv_tol=1e-6

            total_energy=m.kernel()
            m.analyze()
            exit(0)

'''

name=['2X','4Delta','4Phi','2Y','4SigmaP','2Delta','4SigmaM']
mol_spin={'2X':1,'4Delta':3,'4Phi':3,'2Y':1,'4SigmaP':3,'2Delta':1,'4SigmaM':3}
symm_dict={}
symm_dict['2X']=     {'A1' :(5,5), 'E1x': (3,2), 'E1y': (3,3), 'E2x': (1,1), 'E2y': (1,1)} #Ground state
symm_dict['4Pi']=    {'A1' :(6,4), 'E1x': (3,2), 'E1y': (3,3), 'E2x': (1,1), 'E2y': (1,1)} #3sigma 2pi^3 5sigma
symm_dict['4Delta']= {'A1' :(6,5), 'E1x': (3,1), 'E1y': (3,3), 'E2x': (1,1), 'E2y': (1,1)} #1pi^3 2pi^3 5sigma
symm_dict['4Phi']=   {'A1' :(6,5), 'E1x': (3,2), 'E1y': (3,3), 'E2x': (1,0), 'E2y': (1,1)} #1delta^3 2pi^3 5sigma

symm_dict['2Y']=     {'A1' :(5,4), 'E1x': (3,3), 'E1y': (3,3), 'E2x': (1,1), 'E2y': (1,1)} #4sigma 2pi^4
symm_dict['4SigmaP']={'A1' :(6,3), 'E1x': (3,3), 'E1y': (3,3), 'E2x': (1,1), 'E2y': (1,1)} #3sigma 4sigma 5sigma
symm_dict['2Delta']= {'A1' :(5,5), 'E1x': (3,3), 'E1y': (3,3), 'E2x': (1,0), 'E2y': (1,1)} #1delta^3 4sigma2 2pi^4
symm_dict['4SigmaM']={'A1' :(6,5), 'E1x': (3,2), 'E1y': (3,2), 'E2x': (1,1), 'E2y': (1,1)} #2pi^2 5sigma

datacsv={}
for nm in ['molecule','bond-length','charge','spin','method','basis','pseudopotential','totalenergy',
           'totalenergy-stocherr','totalenergy-syserr','pyscf-version']:
  datacsv[nm]=[]

for run in name:
  for r in [1.963925]:
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
            mol.spin=mol_spin[run]
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
      
              m.chkfile=el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol.spin)+"_"+method+"_"+run+".chk"
              m.irrep_nelec = symm_dict[run]
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
              
              dm=m.from_chk(el+'vdz'+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol.spin)+"_"+method+"_"+run+".chk")
              m.chkfile=el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol.spin)+"_"+method+"_"+run+".chk"
              m.irrep_nelec = symm_dict[run]
              m.max_cycle=100
              m = addons.remove_linear_dep_(m)
              m.conv_tol=1e-6
              total_energy=m.kernel(dm)
              m.analyze()

            datacsv['molecule'].append(run)
            datacsv['bond-length'].append(r)
            datacsv['charge'].append(charge)
            datacsv['spin'].append(mol.spin)
            datacsv['method'].append(method)
            datacsv['basis'].append(basis)
            datacsv['pseudopotential'].append('trail')
            datacsv['totalenergy'].append(total_energy)
            datacsv['totalenergy-stocherr'].append(0.0)
            datacsv['totalenergy-syserr'].append(0.0)
            datacsv['pyscf-version'].append('new')
            pd.DataFrame(datacsv).to_csv("cuo_exp.csv",index=False)
'''
