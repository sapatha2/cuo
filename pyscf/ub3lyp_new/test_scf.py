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

S=1
symm_dict=[
{'A1':(5,5),'E1x':(3,3),'E1y':(3,2),'E2x':(1,1),'E2y':(1,1)}
]

'''
for run in [0]:
  for basis in ['vdz','vtz']:
    mol=gto.Mole()
    mol.ecp={}
    mol.basis={}
    for e in ['Cu','O']:
      mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
      mol.basis[e]=gto.basis.parse(df[e][basis])
    mol.charge=0
    mol.spin=S
    mol.build(atom="Cu 0. 0. 0.; O 0. 0. %g"%(r),verbose=4,symmetry=True)

    m=UKS(mol)
    m.xc=xc
    
    m.irrep_nelec = symm_dict[run]
    m.max_cycle=100
    m = addons.remove_linear_dep_(m)
    m.conv_tol=1e-5
    m.diis=scf.ADIIS()

    if basis=='vdz':
      m.chkfile='vdz_'+str(run)+'_'+str(S)+'.chk'
      total_energy=m.kernel()
    
    else:
      dm=m.from_chk('vdz_'+str(run)+'_'+str(S)+'.chk')
      m.chkfile='vtz_'+str(run)+'_'+str(S)+'.chk'
      total_energy=m.kernel(dm)

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
    pd.DataFrame(datacsv).to_csv("scf.csv",index=False)
'''

basis='vtz'
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
m.diis = scf.ADIIS()

#MOM deltaSCF
mo0 = scf.chkfile.load('vtz_0_1.chk','scf/mo_coeff')
occ = scf.chkfile.load('vtz_0_1.chk','scf/mo_occ')
#---------------
occ[1][7]=0
occ[1][12]=1
#---------------

dm = m.make_rdm1(mo0, occ)
m = scf.addons.mom_occ(m, mo0, occ)
#m.chkfile='vtz_'+str(run)+'_'+str(S)+'_mom.chk'

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
