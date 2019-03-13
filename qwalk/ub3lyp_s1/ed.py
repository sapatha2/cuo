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
  ed,epi,epz,tpi,tdz,tsz,tds=parms
  es=0

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

def h2_IAO(Jsd,Us):
  #Jsi = 0.25*(ns_u - ns_d)*(ni_u - ni_d) + 
  #0.5*(cs_u^+ cs_d ci_d^+ ci_u + cs_d^+ cs_u ci_u^+ ci_d)

  #PYSCF ordering: 2rdm[p,r,q,s]_x,x' = <cp_x^+ cq_x'^+ cs_x' cr_x>
  h2=np.zeros((3,9,9,9,9))
  s=8 #index for 4s orbital
  for i in [0,1,2,3,6]:
    h2[0][s,s,i,i]=0.25
    h2[2][s,s,i,i]=0.25
    
    h2[1][i,i,s,s]=-0.25
    h2[1][s,s,i,i]=-0.25
    h2[1][s,i,i,s]=-0.50
    h2[1][i,s,s,i]=-0.50
  h2*=Jsd
  h2[1][s,s,s,s]=Us
  return h2

def ED(parms, nroots, norb, nelec):
  h1=h1_moToIAO(parms[:-2],printvals=True)
  h1=np.array([h1,h1])  #SINGLE PARTICLE CHECK IS OK!
  h2=h2_IAO(parms[-2],parms[-1])

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

if __name__=='__main__':
  nroots=20
  norb=9
  #parms=(-2.9092,-0.6912,-2.0957,0.8289,0,0,0,-0.5530,0)
  parms=(-3.0903,-0.7501,-1.6424,0.6930,0,0,0,-0.4296,0)
  #parms=(-3.0314,-0.6640,-1.5162,0.6780,0,0,0,0,0)

  nelec=(8,7)
  res1=ED(parms,nroots,norb,nelec)

  nelec=(9,6)
  res3=ED(parms,nroots,norb,nelec)

  E = res1[0]
  n_occ_u = res1[2]
  n_occ_d = res1[3]
  Sz = np.ones(len(E))*0.5
  n_3d_u = n_occ_u[:,0] + n_occ_u[:,1] + n_occ_u[:,2] + n_occ_u[:,3] + n_occ_u[:,6]
  n_2ppi_u = n_occ_u[:,4] + n_occ_u[:,5]
  n_2pz_u = n_occ_u[:,7]
  n_4s_u = n_occ_u[:,8]
  n_3d_d = n_occ_d[:,0] + n_occ_d[:,1] + n_occ_d[:,2] + n_occ_d[:,3] + n_occ_d[:,6]
  n_2ppi_d = n_occ_d[:,4] + n_occ_d[:,5]
  n_2pz_d = n_occ_d[:,7]
  n_4s_d = n_occ_d[:,8]
  df = pd.DataFrame({'E':E,'Sz':Sz,'n_3d_u':n_3d_u,'n_2pz_u':n_2pz_u,'n_2ppi_u':n_2ppi_u,'n_4s_u':n_4s_u,
  'n_3d_d':n_3d_d,'n_2pz_d':n_2pz_d,'n_2ppi_d':n_2ppi_d,'n_4s_d':n_4s_d})
  
  E = res3[0]
  n_occ_u = res3[2]
  n_occ_d = res3[3]
  Sz = np.ones(len(E))*1.5
  n_3d_u = n_occ_u[:,0] + n_occ_u[:,1] + n_occ_u[:,2] + n_occ_u[:,3] + n_occ_u[:,6]
  n_2ppi_u = n_occ_u[:,4] + n_occ_u[:,5]
  n_2pz_u = n_occ_u[:,7]
  n_4s_u = n_occ_u[:,8]
  n_3d_d = n_occ_d[:,0] + n_occ_d[:,1] + n_occ_d[:,2] + n_occ_d[:,3] + n_occ_d[:,6]
  n_2ppi_d = n_occ_d[:,4] + n_occ_d[:,5]
  n_2pz_d = n_occ_d[:,7]
  n_4s_d = n_occ_d[:,8]
  df = pd.concat((df,pd.DataFrame({'E':E,'Sz':Sz,'n_3d_u':n_3d_u,'n_2pz_u':n_2pz_u,'n_2ppi_u':n_2ppi_u,'n_4s_u':n_4s_u,
  'n_3d_d':n_3d_d,'n_2pz_d':n_2pz_d,'n_2ppi_d':n_2ppi_d,'n_4s_d':n_4s_d})),axis=0)

  df['E']-=min(df['E'])
  print(df.sort_values(by=['E']))
  #plt.plot(df['E'][df['Sz']==0.5],'go')
  #plt.plot(df['E'][df['Sz']==1.5],'ro')
  df['n_4s']=df['n_4s_u']+df['n_4s_d']
  df['n_3d']=df['n_3d_u']+df['n_3d_d']
  df['n_2ppi']=df['n_2ppi_u']+df['n_2ppi_d']
  df['n_2pz']=df['n_2pz_u']+df['n_2pz_d']
  sns.pairplot(df,vars=['E','n_4s','n_3d','n_2ppi','n_2pz'],hue='Sz')
  plt.savefig('ed_pairplot.pdf',bbox_inches='tight')
