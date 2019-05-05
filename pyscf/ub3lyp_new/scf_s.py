#PySCF input file for CuO calculations 
import json
from pyscf import gto,scf,mcscf, fci,lo,ci,cc,lib
from pyscf.scf import ROHF,ROKS,UHF,UKS, addons
import numpy as np
import pandas as pd

S=1
symm_dict = [
#{'A1':(5,4),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #(z,z -> pi, s)
{'A1':(5,4),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #(z,z -> pi, s)
#{'A1':(5,4),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #(z,z -> pi, s)

{'A1':(5,4),'E1x':(3,3),'E1y':(3,3),'E2x':(1,1),'E2y':(1,1)}, #(dz2 -> pi)
]

chk0 = [
#'../ub3lyp_full/Cuvtz_r1.725_s1_UB3LYP_0.chk', #E = -213.439445239291  <S^2> = 0.78520818  2S+1 = 2.0349036 
'../ub3lyp_full/Cuvtz_r1.725_s1_UB3LYP_3.chk', #E = -213.459052532939  <S^2> = 1.3510888  2S+1 = 2.5306828
#'../ub3lyp_full/Cuvtz_r1.725_s1_UB3LYP_4.chk', #E = -213.440520116249  <S^2> = 0.99639784  2S+1 = 2.2328438

'../ub3lyp_full/Cuvtz_r1.725_s1_UB3LYP_0.chk', #E = -213.476203175783  <S^2> = 1.1669561  2S+1 = 2.3807193
]

excit = []

df=json.load(open("trail.json"))
r = 1.725 
xc = 'B3LYP'
datacsv={}
for nm in['run','method','basis','pseudopotential','bond-length','S','E','conv']:
  datacsv[nm]=[]

basis='vtz'

for run in range(len(chk0)):
  run=0
  mol=gto.Mole()
  mol.ecp={}
  mol.basis={}
  for e in ['Cu','O']:
    mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
    mol.basis[e]=gto.basis.parse(df[e][basis])
  mol.charge = 0
  mol.spin = S
  mol.build(atom="Cu 0. 0. 0.; O 0. 0. %g"%(r), verbose=4, symmetry=True)

  m = UKS(mol)
  m.xc = xc

  m.irrep_nelec = symm_dict[run]
  m.max_cycle = 100
  m = addons.remove_linear_dep_(m)
  m.conv_tol = 1e-5
  #m.diis = scf.ADIIS()

  #MOM deltaSCF
  mo0 = scf.chkfile.load(chk0,'scf/mo_coeff')
  occ = scf.chkfile.load(chk0,'scf/mo_occ')

  #---------------
  #State 1: pz, pz -> pi, 4s using |GS> 
  #occ[0][11]=0
  #occ[0][13]=1
  #occ[1][10]=0
  #occ[1][12]=1

  #MAP excitations
  #---------------

  dm = m.make_rdm1(mo0, occ)
  m = scf.addons.mom_occ(m, mo0, occ)
  m.chkfile='Cuvtz_r1.725_s1_UB3LYP_'+str(run)+'MOM.chk'

  total_energy = m.kernel(dm)
  m.analyze()
  assert(np.sum(m.mo_occ)==25)

  datacsv['run'].append(run)
  datacsv['bond-length'].append(r)
  datacsv['S'].append(S)
  datacsv['method'].append('UB3LYP')
  datacsv['basis'].append(basis)
  datacsv['pseudopotential'].append('trail')
  datacsv['E'].append(total_energy)
  datacsv['conv'].append(m.converged)
  pd.DataFrame(datacsv).to_csv("cuo_MOM.csv",index=False)
