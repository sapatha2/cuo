#Exact diagonalization routine in pyscf
import numpy as np
import scipy.linalg
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS, UKS
from functools import reduce
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 

#Get 1-body parameters in IAO representation
def h1_moToIAO(printvals=False):
  #LOAD IN IAOS
  act_iao=[5,9,6,8,11,12,7,13,1]
  iao=np.load('../../pyscf/ub3lyp_full/b3lyp_iao_b.pickle')
  iao=iao[:,act_iao]
  
  #LOAD IN MOS
  act_mo=[5,6,7,8,9,10,11,12,13]
  chkfile='../../pyscf/ub3lyp_full/Cuvtz_r1.725_s1_UB3LYP_0.chk'
  mol=lib.chkfile.load_mol(chkfile)
  m=UKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
 
  print(m.mo_energy[0][act_mo]*27.2114)
  print(m.mo_energy[1][act_mo]*27.2114)

  for spin in [0]:
    mo=m.mo_coeff[spin][:,act_mo]
    s=m.get_ovlp()

    #IAO ordering: ['del','del','yz','xz','x','y','z2','z','s'] 
    #MO ordering:  dxz, dyz, dz2, delta, delta, px, py, pz, 4s
    e=np.diag(m.mo_energy[spin][act_mo])*27.2114

    mo_to_iao = reduce(np.dot,(mo.T,s,iao))
    e = reduce(np.dot,(mo_to_iao.T,e,mo_to_iao))
    e[np.abs(e)<1e-10]=0
    e=(e+e.T)/2

  return e

def h2_IAO(Jcu,Us):
  #Jsi = 0.25*(ns_u - ns_d)*(ni_u - ni_d) + 
  #0.5*(cs_u^+ cs_d ci_d^+ ci_u + cs_d^+ cs_u ci_u^+ ci_d)

  #PYSCF ordering: 2rdm[p,r,q,s]_x,x' = <cp_x^+ cq_x'^+ cs_x' cr_x>
  h2=np.zeros((3,9,9,9,9))
  index=[0,1,2,3,6,8]
  #for p in range(len(index)):
  #  for j in range(p+1,len(index)):
  j=5
  for p in range(len(index)-1):
      s=index[p]
      i=index[j]
      h2[0][s,s,i,i]=0.25
      h2[2][s,s,i,i]=0.25
      
      h2[1][i,i,s,s]=-0.25
      h2[1][s,s,i,i]=-0.25
      h2[1][s,i,i,s]=-0.50
      h2[1][i,s,s,i]=-0.50
  h2*=Jcu
  h2[1][8,8,8,8]=Us
  return h2

def ED(nroots, norb, nelec):
  h1=h1_moToIAO(printvals=True)
  h1=np.array([h1,h1])
  h2=h2_IAO(0,0)

  #FCI Broken (Based off of pyscf/fci/direct_uhf.py __main__)
  mol = gto.Mole()
  cis = fci.direct_uhf.FCISolver(mol)
  eri_aa = ao2mo.restore(1, h2[0], norb)
  eri_ab = ao2mo.restore(1, h2[1], norb)
  eri_bb = ao2mo.restore(1, h2[2], norb)
  eri = (eri_aa, eri_ab, eri_bb)

  e, ci = fci.direct_uhf.kernel(h1, eri, norb, nelec, nroots=nroots)
  
  #Generate vector of number occupations 
  #['del','del','yz','xz','x','y','z2','z','s']
  n_occ_u = []
  n_occ_d = []
  for i in range(nroots):
    dm=cis.make_rdm1s(ci[i],norb,nelec)
    n_occ_u.append(np.diag(dm[0]))
    n_occ_d.append(np.diag(dm[1]))
  return e, ci, np.array(n_occ_u), np.array(n_occ_d)

def ED_uks_up():
  norb=9

  full_df=None
  nelec=(8,7)
  nroots=14
  res1=ED(nroots,norb,nelec)

  nelec=(9,6)
  nroots=6
  res3=ED(nroots,norb,nelec)

  E = res1[0]
  n_occ = res1[2]+res1[3]
  Sz = np.ones(len(E))*0.5
  n_3d = n_occ[:,0] + n_occ[:,1] + n_occ[:,2] + n_occ[:,3] + n_occ[:,6]
  n_2ppi = n_occ[:,4] + n_occ[:,5]
  n_2pz = n_occ[:,7]
  n_4s = n_occ[:,8]
  df = pd.DataFrame({'E':E,'Sz':Sz,'n_3d':n_3d,'n_2pz':n_2pz,'n_2ppi':n_2ppi,'n_4s':n_4s})
  
  E = res3[0]
  n_occ = res3[2]+res3[3]
  Sz = np.ones(len(E))*1.5
  n_3d = n_occ[:,0] + n_occ[:,1] + n_occ[:,2] + n_occ[:,3] + n_occ[:,6]
  n_2ppi = n_occ[:,4] + n_occ[:,5]
  n_2pz = n_occ[:,7]
  n_4s = n_occ[:,8]
  df = pd.concat((df,pd.DataFrame({'E':E,'Sz':Sz,'n_3d':n_3d,'n_2pz':n_2pz,'n_2ppi':n_2ppi,'n_4s':n_4s})),axis=0)

  df['E']-=min(df['E'])
  df['eig']=np.arange(df.shape[0])
  return df
