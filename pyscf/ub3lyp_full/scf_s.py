#PySCF input file for CuO calculations 
import json
from pyscf import gto,scf,mcscf, fci,lo,ci,cc,lib
from pyscf.scf import ROHF,ROKS,UHF,UKS, addons
import numpy as np
import pandas as pd

df=json.load(open("trail.json"))
charge=0
######################################################################
'''
S=1
symm_dict=[
#4s1 states-----------------------------------------------------------
#3d10 sector
{'A1':(5,5),'E1x':(3,3),'E1y':(3,2),'E2x':(1,1),'E2y':(1,1)}, #GS 
{'A1':(6,5),'E1x':(3,3),'E1y':(2,2),'E2x':(1,1),'E2y':(1,1)}, #(pi -> s)  
{'A1':(6,5),'E1x':(3,2),'E1y':(2,3),'E2x':(1,1),'E2y':(1,1)}, #(pi -> s) 
{'A1':(5,4),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #(z -> pi)
{'A1':(6,4),'E1x':(3,3),'E1y':(2,3),'E2x':(1,1),'E2y':(1,1)}, #(z -> s) 

#3d9 sector
{'A1':(5,5),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,0)}, #(dd -> pi) 
{'A1':(6,5),'E1x':(3,3),'E1y':(2,3),'E2x':(1,1),'E2y':(1,0)}, #(dd -> s) -- Spin pairs 
{'A1':(6,5),'E1x':(3,3),'E1y':(3,2),'E2x':(1,1),'E2y':(0,1)}, #(dd -> s) -- 
{'A1':(5,6),'E1x':(3,3),'E1y':(3,1),'E2x':(1,1),'E2y':(1,1)}, #(dpi -> s)
{'A1':(6,4),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(0,1)}, #(dd,z -> pi,s)

#4s2 states-----------------------------------------------------------
#3d10 sector
{'A1':(6,6),'E1x':(3,2),'E1y':(2,2),'E2x':(1,1),'E2y':(1,1)}, #(2*pi -> 2*s)

#3d9 sector
{'A1':(6,6),'E1x':(3,3),'E1y':(2,2),'E2x':(1,1),'E2y':(1,0)}, #(dd,pi -> s,s)
{'A1':(6,6),'E1x':(3,3),'E1y':(2,1),'E2x':(1,1),'E2y':(1,1)}, #(dpi,pi -> s,s)
{'A1':(6,6),'E1x':(3,2),'E1y':(2,3),'E2x':(1,1),'E2y':(1,0)}, #(dd,pi -> s,s)
{'A1':(6,6),'E1x':(3,1),'E1y':(2,3),'E2x':(1,1),'E2y':(1,1)}, #(dpi,pi-> s,s)
{'A1':(6,6),'E1x':(3,2),'E1y':(3,2),'E2x':(1,1),'E2y':(0,1)}, #(dd,pi -> s,s)
]
'''
#MIRROR (FOR IAOS)
S=1
symm_dict=[
#4s1 states-----------------------------------------------------------
#3d10 sector
{'A1':(5,5),'E1x':(3,2),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #GS 
{'A1':(6,5),'E1x':(2,2),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #(pi -> s)  
{'A1':(6,5),'E1x':(2,3),'E1y':(3,2),'E2x':(1,1),'E2y':(1,1)}, #(pi -> s) 
{'A1':(5,4),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #(z -> pi)
{'A1':(6,4),'E1x':(2,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #(z -> s) 

#3d9 sector
{'A1':(5,5),'E1x':(3,3),'E1y':(3,3),'E2x':(1,0),'E2y':(1,1)}, #(dd -> pi) 
{'A1':(6,5),'E1x':(2,3),'E1y':(3,3),'E2x':(1,0),'E2y':(1,1)}, #(dd -> s) -- Spin pairs 
{'A1':(6,5),'E1x':(3,2),'E1y':(3,3),'E2x':(0,1),'E2y':(1,1)}, #(dd -> s) -- 
{'A1':(5,6),'E1x':(3,1),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #(dpi -> s)
{'A1':(6,4),'E1x':(3,3),'E1y':(3,3),'E2x':(0,1),'E2y':(1,1)}, #(dd,z -> pi,s)

#4s2 states-----------------------------------------------------------
#3d10 sector
{'A1':(6,6),'E1x':(2,2),'E1y':(3,2),'E2x':(1,1),'E2y':(1,1)}, #(2*pi -> 2*s)

#3d9 sector
{'A1':(6,6),'E1x':(2,2),'E1y':(3,3),'E2x':(1,0),'E2y':(1,1)}, #(dd,pi -> s,s)
{'A1':(6,6),'E1x':(2,1),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #(dpi,pi -> s,s)
{'A1':(6,6),'E1x':(2,3),'E1y':(3,2),'E2x':(1,0),'E2y':(1,1)}, #(dd,pi -> s,s)
{'A1':(6,6),'E1x':(2,3),'E1y':(3,1),'E2x':(1,1),'E2y':(1,1)}, #(dpi,pi-> s,s)
{'A1':(6,6),'E1x':(3,2),'E1y':(3,2),'E2x':(0,1),'E2y':(1,1)}, #(dd,pi -> s,s)
]
######################################################################
'''
S=3
symm_dict=[
#4s1 states-----------------------------------------------------------
#3d10 sector
{'A1':(6,5),'E1x':(3,2),'E1y':(3,2),'E2x':(1,1),'E2y':(1,1)}, #dn - 2pz occupied
{'A1':(6,4),'E1x':(3,3),'E1y':(3,2),'E2x':(1,1),'E2y':(1,1)}, #dn - 2ppi occupied
#3d9 sector
{'A1':(6,5),'E1x':(3,3),'E1y':(3,2),'E2x':(1,1),'E2y':(1,0)}, #delta, dn - pz, px
{'A1':(6,4),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,0)}, #delta, dn - px, py
{'A1':(6,5),'E1x':(3,3),'E1y':(3,1),'E2x':(1,1),'E2y':(1,1)}, #pi, dn - pz, px
{'A1':(6,3),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #dz2, dn - px, py

#4s2 states-----------------------------------------------------------
#3d9 sector
#{'A1':(6,6),'E1x':(3,2),'E1y':(3,2),'E2x':(1,1),'E2y':(1,0)}, 
#{'A1':(6,6),'E1x':(3,2),'E1y':(3,1),'E2x':(1,1),'E2y':(1,1)}, 
]
'''
'''
#MIRROR (FOR IAOS)
S=3
symm_dict=[
#4s1 states-----------------------------------------------------------
#3d10 sector
{'A1':(6,5),'E1x':(3,2),'E1y':(3,2),'E2x':(1,1),'E2y':(1,1)}, #dn - 2pz occupied
{'A1':(6,4),'E1x':(3,2),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #dn - 2ppi occupied
#3d9 sector
{'A1':(6,5),'E1x':(3,2),'E1y':(3,3),'E2x':(1,0),'E2y':(1,1)}, #delta, dn - pz, px
{'A1':(6,4),'E1x':(3,3),'E1y':(3,3),'E2x':(1,0),'E2y':(1,1)}, #delta, dn - px, py
{'A1':(6,5),'E1x':(3,1),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #pi, dn - pz, px
{'A1':(6,3),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #dz2, dn - px, py
]
'''
######################################################################

datacsv={}

for nm in['run','method','basis','pseudopotential','bond-length','S','E','conv']:
  datacsv[nm]=[]

for run in np.arange(10,16):
  for r in [1.725]:
    for method in ['UB3LYP']:
      for basis in ['vdz','vtz']:
        for el in ['Cu']:
          molname=el+'O'
          mol=gto.Mole()

          mol.ecp={}
          mol.basis={}
          for e in [el,'O']:
            mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
            mol.basis[e]=gto.basis.parse(df[e][basis])
          mol.charge=charge
          mol.spin=S
          mol.build(atom="%s 0. 0. 0.; O 0. 0. %g"%(el,r),verbose=4,symmetry=True)
         
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

          if basis=='vdz':
            #m=m.newton()
            m.chkfile=el+basis+"_r"+str(r)+"_s"+str(S)+"_"+method+"_"+str(run)+"mirror.chk"
            m.irrep_nelec = symm_dict[run]
            m.max_cycle=100
            m = addons.remove_linear_dep_(m)
            m.conv_tol=1e-5
            m.diis=scf.ADIIS()
            total_energy=m.kernel()
            
            #Compute the Mulliken orbital occupancies...
            m.analyze()
            #m.stability(external=True)
            assert(np.sum(m.mo_occ)==25)
          
          #Once we get past the vdz basis, just read-in the existingmirror.chk file...
          else:
            dm=m.from_chk(el+'vdz'+"_r"+str(r)+"_s"+str(S)+"_"+method+"_"+str(run)+"mirror.chk")
            m.chkfile=el+basis+"_r"+str(r)+"_s"+str(S)+"_"+method+"_"+str(run)+"mirror.chk"
            m.irrep_nelec = symm_dict[run]
            m.max_cycle=100
            m = addons.remove_linear_dep_(m)
            m.conv_tol=1e-5
            m.diis=scf.ADIIS()
            total_energy=m.kernel(dm)

            m.analyze()
            #m.stability(external=True)
            assert(np.sum(m.mo_occ)==25)

          datacsv['run'].append(run)
          datacsv['bond-length'].append(r)
          datacsv['S'].append(S)
          datacsv['method'].append(method)
          datacsv['basis'].append(basis)
          datacsv['pseudopotential'].append('trail')
          datacsv['E'].append(total_energy)
          datacsv['conv'].append(m.converged)
          pd.DataFrame(datacsv).to_csv("cuo_do.csv",index=False)
