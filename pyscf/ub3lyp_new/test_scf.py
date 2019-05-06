#PySCF input file for CuO calculations 
import json
from pyscf import gto,scf,mcscf, fci,lo,ci,cc,lib
from pyscf.scf import ROHF,ROKS,UHF,UKS, addons
import numpy as np
import pandas as pd

df=json.load(open("trail.json"))
r = 1.725 
xc = 'B3LYP'
basis='vtz'
datacsv={}
for nm in['run','method','basis','pseudopotential','bond-length','S','E','conv']:
  datacsv[nm]=[]

#Up: 5 [dy dz dx dd dd py pz px 4s] 13
#Dn: 5 [dz dx dy dd dd pz px py 4s] 13
#|GS>

#Up: 5 [dy dz dx dd dd pz px 4s py] 13
#Dn: 5 [dd dx dy dz pz px py dd 4s] 13
#|1>

#Up:  
#Dn: 5 [dd dy dx dz py pz px dd 4s] 13
#|6>

#Up: 5 [dz dx dy dd dd pz px 4s py] 13
#Dn: 5 [dy dx dd dd dz py px pz 4s] 13
#|4> 

#Up: 
#Dn: 5 [dz dd dd dx dy pz px py 4s] 13 
#|GS 3/2>

#Up: 
#Dn: 5 [dy dx dd dd dz px pz py 4s] 
#|1 3/2>

#Up: 
#Dn: 5 [dx dy dd dd px py dz pz 4s] 
#|5 3/2>

'''
S=[1,1,1,3]

chk0=[
'../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_0.chk',
'../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_0.chk',
'../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_0.chk',
'../ub3lyp_full/Cu'+str(basis)+'_r1.725_s3_UB3LYP_5.chk',
]

excit_list = [
[[1,10,12],[0,11,13]],  #(z,z -> pi, s) (2,0,1) OK!  [4s up, dz dn, pz dn]
[[1,7,12]],             #(dy -> py)     (3,4)   OK!  [4s up, pz dn, 3dy dn, 2py up]
[[1,5,12]],             #(dz -> py)                  [4s up, dz dn, pz dn]
[[1,10,12]],             #(dz -> 4s)               
]# Build this
'''

S=[1,1,3]

chk0=[
'../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_0.chk',
'../ub3lyp_full/Cu'+str(basis)+'_r1.725_s1_UB3LYP_0.chk',
'../ub3lyp_full/Cu'+str(basis)+'_r1.725_s3_UB3LYP_5.chk',
]

excit_list = [
[[1,10,12],[0,11,13]],
[[1,7,12]],             
[[1,10,12]],            
]# Build this

for run in np.arange(3):
  mol=gto.Mole()
  mol.ecp={}
  mol.basis={}
  for e in ['Cu','O']:
    mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
    mol.basis[e]=gto.basis.parse(df[e][basis])
  mol.charge = 0
  mol.spin = S[run]
  mol.build(atom="Cu 0. 0. 0.; O 0. 0. %g"%(r), verbose=4)

  m = UKS(mol)
  m.xc = xc
  m.max_cycle = 100
  m = addons.remove_linear_dep_(m)
  m.conv_tol = 1e-5

  #MOM deltaSCF
  mo0 = scf.chkfile.load(chk0[run],'scf/mo_coeff')
  occ = scf.chkfile.load(chk0[run],'scf/mo_occ')

  #---------------
  if(len(excit_list[run])>0):
    for z in excit_list[run]:
      occ[z[0]][z[1]]=0
      occ[z[0]][z[2]]=1
      dm = m.make_rdm1(mo0, occ)
      m = scf.addons.mom_occ(m, mo0, occ)
  else: 
    dm=m.from_chk(chk0)

  m.chkfile='Cuvtz_r1.725_s'+str(S[run])+'_UB3LYP_'+str(run+11)+'.chk'

  total_energy = m.kernel(dm)
  m.analyze()
  assert(np.sum(m.mo_occ)==25)

  datacsv['run'].append(run+11)
  datacsv['bond-length'].append(r)
  datacsv['S'].append(S)
  datacsv['method'].append('UB3LYP')
  datacsv['basis'].append(basis)
  datacsv['pseudopotential'].append('trail')
  datacsv['E'].append(total_energy)
  datacsv['conv'].append(m.converged)
  pd.DataFrame(datacsv).to_csv("cuo_u.csv",index=False)
