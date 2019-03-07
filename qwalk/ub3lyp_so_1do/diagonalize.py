import numpy as np 
import matplotlib.pyplot as plt
from pyscf import lib
from pyscf.scf import ROKS
from functools import reduce 

def diagonalize(parms):
  e4s,e3d,epi,ez,tpi,tds,tdz,tsz=parms
  H=np.diag([e3d,e3d,e3d,e3d,e3d,epi,epi,ez,e4s])
  
  #e4s,e3dpi,e3dz2,e3dd,epi,ez,tpi,tds,tdz,tsz=parms
  #H=np.diag([e3dpi,e3dpi,e3dz2,e3dd,e3dd,epi,epi,ez,e4s])

  #dx, dy, dz2, dd, px, py, pz, 4s
  H[[0,1,5,6],[5,6,0,1]]=tpi
  H[[2,7],[7,2]]=tdz
  H[[7,8],[8,7]]=tsz
  H[[2,8],[8,2]]=-tds

  w,vr=np.linalg.eigh(H)
  return(w-w[0])

def new_gs(parms):
  print(parms)
  e4s,e3d,epi,ez,tpi,tds,tdz,tsz=parms
  
  #dx, dy, dz2, dd, px, py, pz, 4s
  H=np.diag([e3d,e3d,e3d,e3d,e3d,epi,epi,ez,e4s])
  H[[0,1,5,6],[5,6,0,1]]=tpi
  H[[2,7],[7,2]]=tdz
  H[[7,8],[8,7]]=tsz
  H[[2,8],[8,2]]=tds

  w,vr=np.linalg.eigh(H)

  from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
  from pyscf.scf import ROHF,UHF,ROKS
  from pyscf2qwalk import print_qwalk_mol
  chkfile="../../pyscf/chk/Cuvtz_r1.725_s1_B3LYP_1.chk"
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  
  m.mo_coeff[:,5:14]=np.dot(m.mo_coeff[:,5:14],vr)
  print_qwalk_mol(mol,m,basename="new_gs/new_gs")

'''
const      -5776.3165      0.297  -1.94e+04      0.000   -5776.900   -5775.733
n_3d          -2.5089      0.025   -102.135      0.000      -2.557      -2.461
n_2ppi        -0.4060      0.024    -17.199      0.000      -0.452      -0.360
n_2pz         -1.9120      0.027    -71.364      0.000      -1.965      -1.859
t_pi           1.0849      0.026     41.361      0.000       1.033       1.136
sigU           4.2194      0.054     78.338      0.000       4.114       4.325
'''

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
ed,epi,epz,es,tpi,tdz,tsz,tds,Us=(-2.5089,-0.4060,-1.9120,0,1.0849,0,0,0,4.2194)
e=np.diag([ed,ed,ed,ed,ed,epi,epi,epz,es])
e[[0,1,5,6],[5,6,0,1]]=tpi
e[[2,7],[7,2]]=tdz
e[[8,7],[7,8]]=tsz
e[[8,2],[2,8]]=tds

mo_to_iao = reduce(np.dot,(mo.T,s,iao))
e = reduce(np.dot,(mo_to_iao.T,e,mo_to_iao))
e[np.abs(e)<1e-10]=0

'''
plt.matshow(e,vmin=-3,vmax=3,cmap=plt.cm.bwr)
plt.xticks(np.arange(9),['del','del','yz','xz','x','y','z2','z','s'],rotation=90)
plt.yticks(np.arange(9),['del','del','yz','xz','x','y','z2','z','s'])
plt.show()
'''

'''
edel=e[:2,:2]
epi= e[2:6,2:6]

edel,__=np.linalg.eigh(edel)
epi, __=np.linalg.eigh(epi)
'''

z2=e[6,6] #Occupations
z= e[7,7]
s= e[8,8]
dz=e[6,7] #Hopping
ds=e[6,8]
zs=e[7,8]

#-------------------------------------------------------------------------------------------------------
#Nsigma=3
#Sz=3(-3), (u, u, u)
w=z2+z+s
vr=1
print(w)

#Sz=1(-1), (2, u, .), (2, . u), (u, 2, .), (., 2, u), (u, ., 2), (., u, 2)
#(u, d, u), (d, u, u), (u, u, d)
esig=np.zeros((9,9))
esig[[0,0,0],[7,8,1]]=[ds,-ds,zs]
esig[[1,1],[7,6]]=[dz,-dz]
esig[[2,2,2],[3,6,8]]=[ds,zs,-zs]
esig[[3,3],[6,7]]=[ds,-ds]
esig[[4,4,4],[5,8,6]]=[dz,zs,-zs]
esig[[5,5],[8,7]]=[ds,-ds]
esig+=esig.T
esig+=np.diag([2*z2+z,2*z2+s,z2+2*z,2*z+s,z2+2*s+Us,z+2*s+Us,z2+z+s,z2+z+s,z2+z+s])
w,vr=np.linalg.eigh(esig)
print(w)

#-------------------------------------------------------------------------------------------------------
#Nsigma=4
#Sz=2(-2) (2, u, u), (u, 2, u), (u ,u, 2)
esig=np.array([[2*z2+z+s,-dz,-ds],[-dz,z2+2*z*s,-zs],[-ds,-zs,z2+z+2*s+Us]])
w,vr=np.linalg.eigh(esig)
print(w)

#Sz=0 (2, 2, .), (2, ., 2), (., 2, 2), 
#(2, u, d), (2, d, u), (u, 2, d), (d, 2, u)
#(u, d, 2), (d, u, 2)
esig=np.zeros((9,9))
esig[[0,0,0,0],[6,5,4,3]]=[ds,-ds,zs,-zs]
esig[[1,1,1,1],[7,8,5,6]]=[dz,-dz,ds,-ds]
esig[[2,2,2,2],[8,7,3,4]]=[dz,-dz,zs,-zs]
esig[[3,3],[5,8]]=[-dz,ds]
esig[[4,4],[6,7]]=[dz,-ds]
esig[[5,6],[7,8]]=[zs,zs]
esig+=esig.T
esig+=np.diag([2*z2+2*z,2*z+2*s+Us,2*z2+2*s+Us,2*z2+z+s,2*z2+z+s,z2+2*z+s,z2+2*z+s,z2+z+2*s,z2+z+2*s])
w,vr=np.linalg.eigh(esig)
print(w)


