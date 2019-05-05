#PySCF input file for CuO calculations 
import json
from pyscf import gto,scf,mcscf, fci,lo,ci,cc,lib
from pyscf.scf import ROHF,ROKS,UHF,UKS, addons
import numpy as np
import pandas as pd

df=json.load(open("trail.json"))
r = 1.725 
xc = 'B3LYP'
datacsv={}
for nm in['run','method','basis','pseudopotential','bond-length','S','E','conv']:
  datacsv[nm]=[]

basis='vtz'

#Up: 5 [dy dz dx dd dd py pz px 4s] 13
#Dn: 5 [dz dx dy dd dd pz px py 4s] 13
chk0='../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_0.chk'

#Up: 5 [dz dx dy dd dd pz px py 4s] 13
#Dn: 5 [dx dy dd dd dz px py pz 4s] 13
#chk0='../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_3.chk'

#Up: 5 [dz dx dy dd dd pz px 4s py] 13
#Dn: 5 [dy dx dd dd dz py px pz 4s] 13
#chk0='../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_4.chk'

#Up: 5 [dy dz dx dd dd py pz px 4s] 13
#Dn: 5 [dz dx dd dd pz px 4s dy py] 13
#chk0='../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_8.chk' 

#Up: 5 [--------------------------] 13
#Dn: 5 [dd dx dy dz pz px py dd 4s] 13
#chk0='../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_5.chk' 

#Up: 5 [dy dz dx dd dd pz px 4s py] 13
#Dn: 5 [dz dx dd dd dy pz px py 4s] 13
#chk0='../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_1.chk' 

#Up: 5 [dx dz dd dd dy px pz 4s py] 13
#Dn: 5 [dy dz dd dd py dx pz px 4s] 13
#chk0='../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_2.chk' 

#Up: 5 [--------------------------] 13
#Dn: 5 [dz dd dd dx dy pz px py 4s] 13
#chk0='../ub3lyp_full/Cu'+str(basis)+'_r1.725_s3_UB3LYP_0.chk'

#Up: 5 [--------------------------] 13
#Dn: 5 [dz dx dd dd pz px py dy 4s] 13
#chk0='../ub3lyp_full/Cu'+str(basis)+'_r1.725_s3_UB3LYP_4.chk'

mol=gto.Mole()
mol.ecp={}
mol.basis={}
for e in ['Cu','O']:
  mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
  mol.basis[e]=gto.basis.parse(df[e][basis])
mol.charge = 0
mol.spin = 1
mol.build(atom="Cu 0. 0. 0.; O 0. 0. %g"%(r), verbose=4)

m = UKS(mol)
m.xc = xc

m.max_cycle = 100
m = addons.remove_linear_dep_(m)
m.conv_tol = 1e-5

#MOM deltaSCF
mo0 = scf.chkfile.load(chk0,'scf/mo_coeff')
occ = scf.chkfile.load(chk0,'scf/mo_occ')

#---------------
occ[1][7]=0
occ[1][12]=1
#---------------

dm = m.make_rdm1(mo0, occ)
m = scf.addons.mom_occ(m, mo0, occ)

m.chkfile='test_mom.chk'

total_energy = m.kernel(dm)
m.analyze()
assert(np.sum(m.mo_occ)==25)

'''
datacsv['run'].append(run)
datacsv['bond-length'].append(r)
datacsv['S'].append(S)
datacsv['method'].append('UB3LYP')
datacsv['basis'].append(basis)
datacsv['pseudopotential'].append('trail')
datacsv['E'].append(total_energy)
datacsv['conv'].append(m.converged)
pd.DataFrame(datacsv).to_csv("scf.csv",index=False)
'''
