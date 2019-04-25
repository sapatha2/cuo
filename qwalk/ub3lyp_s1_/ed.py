#Exact diagonalization routine in pyscf
import numpy as np
import scipy.linalg
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 

#Get 1-body parameters in IAO representation
def h1_moToIAO(parms,printvals=False):
  #LOAD IN IAOS
  act_iao=[5,9,6,8,11,12,7,13,1]
  iao=np.load('../../pyscf/ub3lyp_full/b3lyp_iao_b.pickle')
  iao=iao[:,act_iao]
  
  #LOAD IN MOS
  act_mo=[5,6,7,8,9,10,11,12,13]
  chkfile='../../pyscf/chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  mo=m.mo_coeff[:,act_mo]
  s=m.get_ovlp()

  #IAO ordering: ['del','del','yz','xz','x','y','z2','z','s'] 
  #MO ordering:  dxz, dyz, dz2, delta, delta, px, py, pz, 4s
  es,epi,epz,tpi,tdz,tsz,tds=parms
  ed=0

  e=np.diag([ed,ed,ed,ed,ed,epi,epi,epz,es])
  e[[0,1,5,6],[5,6,0,1]]=tpi
  e[[2,7],[7,2]]=tdz
  e[[8,7],[7,8]]=tsz
  e[[8,2],[2,8]]=tds
  
  if(printvals):
    w,vr=np.linalg.eigh(e)
    print('MO eigenvalues, Jsd=0 ------------------------------------')
    print(w)

  mo_to_iao = reduce(np.dot,(mo.T,s,iao))
  e = reduce(np.dot,(mo_to_iao.T,e,mo_to_iao))
  e[np.abs(e)<1e-10]=0
  e=(e+e.T)/2
 
  if(printvals):
    w,vr=np.linalg.eigh(e)
    print('IAO eigenvalues, Jsd=0 ------------------------------------')
    print(w)

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
  #Generate vector of number occupations 
  #['del','del','yz','xz','x','y','z2','z','s']
  dm_u = []
  dm_d = []
  sigU = []
  sigJ = []
  for i in range(nroots):
    dm2=cis.make_rdm12s(ci[i],norb,nelec)
    dm_u.append(dm2[0][0])
    dm_d.append(dm2[0][1])
    sigU.append(dm2[1][1][8,8,8,8]) #U4s parameter sum
    
    Jsd = 0
    for i in [0,1,2,3,6]:
      Jsd += 0.25*(dm2[1][0][8,8,i,i] + dm2[1][2][8,8,i,i] - dm2[1][1][8,8,i,i] - dm2[1][1][i,i,8,8])-\
             0.5*(dm2[1][1][i,8,8,i] + dm2[1][1][8,i,i,8])
    sigJ.append(Jsd) #Jsd parameter sum
  return e, ci, np.array(dm_u), np.array(dm_d), sigU, sigJ
