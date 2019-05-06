import numpy as np 
import matplotlib.pyplot as plt
from pyscf import lib
from pyscf.scf import ROKS
from functools import reduce 

'''
const      -5776.3165      0.297  -1.94e+04      0.000   -5776.900   -5775.733
n_3d          -2.5089      0.025   -102.135      0.000      -2.557      -2.461
n_2ppi        -0.4060      0.024    -17.199      0.000      -0.452      -0.360
n_2pz         -1.9120      0.027    -71.364      0.000      -1.965      -1.859
t_pi           1.0849      0.026     41.361      0.000       1.033       1.136
sigU           4.2194      0.054     78.338      0.000       4.114       4.325


n_3d          -2.5356      0.025    -99.798      0.000      -2.585      -2.486
n_2ppi        -0.8657      0.049    -17.561      0.000      -0.962      -0.769
n_2pz         -2.0601      0.031    -67.187      0.000      -2.120      -2.000
t_pi           0.8990      0.038     23.659      0.000       0.824       0.973
t_sz           0.8957      0.079     11.337      0.000       0.741       1.051
t_dz          -0.3208      0.047     -6.883      0.000      -0.412      -0.229
sigU           3.4658      0.085     41.013      0.000       3.300       3.632
'''

def diagonalize_3d10(parms,printvals=False):
  #LOAD IN IAOS
  act_iao=[5,9,6,8,11,12,7,13,1]
  iao=np.load('../../pyscf/ub3lyp_full/b3lyp_iao_b.pickle')
  iao=iao[:,act_iao]
  #iao[:,7]*=-1 #CORRECT SIGN ON PZ

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
  ed,epi,epz,tpi,tdz,tsz,tds,Us=parms
  es=0

  e=np.diag([ed,ed,ed,ed,ed,epi,epi,epz,es])
  e[[0,1,5,6],[5,6,0,1]]=tpi
  e[[2,7],[7,2]]=tdz
  e[[8,7],[7,8]]=tsz
  e[[8,2],[2,8]]=tds
  w,vr=np.linalg.eigh(e)
  if(printvals): print('MO eigenvalues, Us=0 ------------------------------------')
  if(printvals): print(w)

  mo_to_iao = reduce(np.dot,(mo.T,s,iao))
  e = reduce(np.dot,(mo_to_iao.T,e,mo_to_iao))
  e[np.abs(e)<1e-10]=0
  e=(e+e.T)/2

  w,vr=np.linalg.eigh(e)
  #plt.matshow(e,vmin=-3,vmax=3,cmap=plt.cm.bwr)
  #plt.xticks(np.arange(9),['del','del','yz','xz','x','y','z2','z','s'],rotation=90)
  #plt.yticks(np.arange(9),['del','del','yz','xz','x','y','z2','z','s'])
  #plt.show()
  #exit(0)

  edel=e[:2,:2]
  epi= e[2:6,2:6]
  esig=e[6:,6:]
  edel,__=np.linalg.eigh(edel)
  epi, __=np.linalg.eigh(epi)
  esig,__=np.linalg.eigh(esig)
  if(printvals): print('IAO eigenvalues, Us=0 ------------------------------------')
  if(printvals): print(edel)
  if(printvals): print(epi)
  if(printvals): print(esig)

  z2=e[6,6] #Occupations
  z= e[7,7]
  s= e[8,8]
  dz=-1*e[6,7] #Hopping, note IAO and GS1 pz are opposite sign
  ds=-1*e[6,8]
  zs=-1*e[7,8]
  #Us=0

  e_3d10=[]
  #-------------------------------------------------------------------------------------------------------
  if(printvals): print('Nsigma=3 ----------------------------------------------------')
  #Nsigma=3
  #Sz=3(-3), (u, u, u)
  w=z2+z+s
  vr=1
  if(printvals): print(w+epi[2]*4)
  e_3d10+=list([w+epi[2]*4])

  #Sz=1(-1), (2, u, .), (2, . u), (u, 2, .), (., 2, u), (u, ., 2), (., u, 2)
  #(u, d, u), (d, u, u), (u, u, d)
  esig=np.zeros((9,9))
  esig[[0,0,0],[7,8,1]]=[ds,-ds,zs]
  esig[[1,1],[7,6]]=[-dz,dz]
  esig[[2,2,2],[3,6,8]]=[ds,-zs,zs]
  esig[[3,3],[6,7]]=[dz,-dz]
  esig[[4,4,4],[5,8,6]]=[dz,zs,-zs]
  esig[[5,5],[8,7]]=[-ds,ds]
  esig[[0,1,3],[2,4,5]]=[-dz,-ds,-zs]
  esig+=esig.T
  esig+=np.diag([2*z2+z,2*z2+s,z2+2*z,2*z+s,z2+2*s+Us,z+2*s+Us,z2+z+s,z2+z+s,z2+z+s])
  w,vr=np.linalg.eigh(esig)
  if(printvals): print(w+epi[2]*4)
  e_3d10+=list(w+epi[2]*4)

  #-------------------------------------------------------------------------------------------------------
  if(printvals): print('Nsigma=4 ----------------------------------------------------')
  #Nsigma=4
  #Sz=2(-2) (2, u, u), (u, 2, u), (u ,u, 2)
  esig=np.array([[2*z2+z+s,-dz,ds],[-dz,z2+2*z+s,-zs],[ds,-zs,z2+z+2*s+Us]])
  w,vr=np.linalg.eigh(esig)
  if(printvals): print(w+epi[2]*3)
  e_3d10+=list(w+epi[2]*3)

  #Sz=0 (2, 2, .), (., 2, 2), (2, ., 2), 
  #(2, u, d), (2, d, u), (u, 2, d), (d, 2, u)
  #(u, d, 2), (d, u, 2)
  esig=np.zeros((9,9))
  esig[[0,0,0,0],[6,5,4,3]]=[-ds,ds,-zs,zs]
  esig[[1,1,1,1],[7,8,5,6]]=[dz,-dz,ds,-ds]
  esig[[2,2,2,2],[8,7,3,4]]=[-dz,dz,zs,-zs]
  esig[[3,3],[5,8]]=[-dz,ds]
  esig[[4,4],[6,7]]=[-dz,ds]
  esig[[5,6],[7,8]]=[-zs,-zs]
  esig+=esig.T
  esig+=np.diag([2*z2+2*z,2*z+2*s+Us,2*z2+2*s+Us,2*z2+z+s,2*z2+z+s,z2+2*z+s,z2+2*z+s,z2+z+2*s+Us,z2+z+2*s+Us])
  w,vr=np.linalg.eigh(esig)
  if(printvals): print(w+epi[2]*3)
  e_3d10+=list(w+epi[2]*3)

  #-------------------------------------------------------------------------------------------------------
  if(printvals): print('Nsigma=5 ----------------------------------------------------')
  #Nsigma=5
  #Sz=1(-1) (2, 2, u), (2, u, 2), (u ,2, 2)
  esig=np.zeros((3,3))
  esig[[0,0,1],[1,2,2]]=[-zs,-ds,-dz]
  esig+=esig.T
  esig+=np.diag([2*z2+2*z+s,2*z2+z+2*s+Us,z2+2*z+2*s+Us])
  w,vr=np.linalg.eigh(esig)
  if(printvals): print(w+epi[2]*2)
  e_3d10+=list(w+epi[2]*2)

  #-------------------------------------------------------------------------------------------------------
  #if(printvals): print('Nsigma=6 ----------------------------------------------------')
  #Nsigma=6
  #Sz=0, (2,2,2)
  #w=2*z2+2*z+2*s+Us
  #vr=1
  #if(printvals): print(w)

  #------------------------------------------------------------------------------------------------------
  e_3d10=np.array(e_3d10)-min(e_3d10)
  return np.array(sorted(e_3d10))

#diagonalize_3d10(parms=[ -3.2895,-0.8015,-1.8085, 0.8379,0,0,0,0])
