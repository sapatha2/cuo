#Exact diagonalization routine in pyscf
import numpy as np
import scipy.linalg
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
from scipy.optimize import linear_sum_assignment

#Get 1-body parameters in IAO representation
def h1_moToIAO(parms,printvals=False):
  #LOAD IN IAOS
  act_iao=[5,9,6,8,11,12,7,13,1]
  iao=np.load('../../../pyscf/ub3lyp_full/b3lyp_iao_b.pickle')
  iao=iao[:,act_iao]
  
  #LOAD IN MOS
  act_mo=[5,6,7,8,9,10,11,12,13]
  chkfile='../../../pyscf/chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  mo=m.mo_coeff[:,act_mo]
  s=m.get_ovlp()

  #IAO ordering: ['del','del','yz','xz','x','y','z2','z','s'] 
  #MO ordering:  dxz, dyz, dz2, delta, delta, px, py, pz, 4s
  #es,edpi,edz2,edd,epi,epz,tpi,tdz,tsz,tds=parms
  #e=np.diag([edpi,edpi,edz2,edd,edd,epi,epi,epz,es])
  
  es,epi,epz,tpi,tdz,tsz,tds=parms
  ed=0 
  e=np.diag([ed,ed,ed,ed,ed,epi,epi,epz,es])
  e[[0,1,5,6],[5,6,0,1]]=tpi
  e[[2,7],[7,2]]=tdz
  e[[8,7],[7,8]]=tsz
  e[[8,2],[2,8]]=tds

  mo_to_iao = reduce(np.dot,(mo.T,s,iao))
  e = reduce(np.dot,(mo_to_iao.T,e,mo_to_iao))
  e[np.abs(e)<1e-10]=0
  e=(e+e.T)/2
  
  return e

def h2_IAO(Jcu,Us,t=False):
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
      h2[0][i,i,s,s]=0.25
      h2[2][s,s,i,i]=0.25
      h2[2][i,i,s,s]=0.25
      
      h2[1][i,i,s,s]=-0.25
      h2[1][s,s,i,i]=-0.25
      h2[1][s,i,i,s]=-0.50
      h2[1][i,s,s,i]=-0.50
  h2*=Jcu
  h2[1][8,8,8,8]=Us

  return h2

def ED(parms, nroots, norb, nelec):
  h1=h1_moToIAO(parms[:-2])
  h1=np.array([h1,h1])  #SINGLE PARTICLE CHECK IS OK!
  h2=h2_IAO(parms[-2],parms[-1])
  
  #plt.matshow(h1[0],vmin=-5,vmax=5,cmap=plt.cm.bwr)
  #plt.show()

  #FCI Broken (Based off of pyscf/fci/direct_uhf.py __main__)
  mol = gto.Mole()
  cis = fci.direct_uhf.FCISolver(mol)
  eri_aa = ao2mo.restore(1, h2[0], norb)
  eri_ab = ao2mo.restore(1, h2[1], norb)
  eri_bb = ao2mo.restore(1, h2[2], norb) #4 and 8 fold identical since we have no complex entries
  eri = (eri_aa, eri_ab, eri_bb)

  e, ci = fci.direct_uhf.kernel(h1, eri, norb, nelec, nroots=nroots)
  return e, ci

if __name__=='__main__':
  norb = 9
  nelec = (8,7)
  nroots = 30
  
  #MINIMAL MODEL
  #v = [0.31,0.19,0,2.16,1.7,0.93,-0.57,0.53,0.88,0.45,-0.6,4.0]
  #EXTENDED MODEL
  v = [0.58,0.08,0,1.2,2.13,1.73,-0.47,1.00,1.26,0.72,-0.4,4.5]
  edz2, edpi, edd, es, eppi, epz, tpi, tdz, tsz, tds, Jsd, Us = v

  h1 = np.zeros((norb,norb))
  h1[2,5] = h1[3,4] = tpi
  h1[6,7] = tdz
  h1[7,8] = tsz 
  h1[6,8] = tds
  h1 += h1.T
  h1[6,6] = edz2
  h1[2,2] = h1[3,3] = edpi
  h1[0,0] = h1[1,1] = edd
  h1[4,4] = h1[5,5] = eppi
  h1[7,7] = epz
  h1[8,8] = es

  h1 = np.array([h1,h1])
  h2 = h2_IAO(v[-2],v[-1])

  mol = gto.Mole()
  cis = fci.direct_uhf.FCISolver(mol)
  eri_aa = ao2mo.restore(1, h2[0], norb)
  eri_ab = ao2mo.restore(1, h2[1], norb)
  eri_bb = ao2mo.restore(1, h2[2], norb) #4 and 8 fold identical since we have no complex entries
  eri = (eri_aa, eri_ab, eri_bb)

  e, ci = fci.direct_uhf.kernel(h1, eri, norb, nelec, nroots=nroots)
  print(np.around(e-min(e),2))
