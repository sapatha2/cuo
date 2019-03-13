import numpy as np 
import matplotlib.pyplot as plt
from functools import reduce 

N=np.zeros((20,20))
T=np.zeros((20,20))
J=np.zeros((20,20))
H=np.zeros((20,20))

#Construct T
T[[2,2,3,3],[5,8,5,8]]=[1,1,-1,-1]
T[6,7]=-1
T[9,10]=1

T[[12,12,13,13],[14,17,14,17]]=[-1,-1,1,1]
T[15,16]=-1
T[18,19]=1
T+=T.T

#Construct J
J[[3,11],[4,13]]=0.5
J+=J.T
J[[0,1,2,3,4,11,12,13],[0,1,2,3,4,11,12,13]]=\
[0.25,0.25,0.25,-0.25,-0.25,-0.25,0.25,-0.25]

#Construct N
ed=-3.53
ep=-0.547
N=np.diag([ed+ep,ed+ep,ed+ep,ed+ep,ed+ep,2*ed,2*ed+ep,ed+2*ep,2*ep,ep,ed,
ed+ep,ed+ep,ed+ep,2*ed,2*ed+ep,ed+2*ep,2*ep,ep,ed])

H=-0.085*T-0.8953*J+N
w,vr=np.linalg.eigh(H)
print(w)
for i in range(len(w)):
  print(vr[:,i])
  Jval=reduce(np.dot,(vr[:,i].T,J,vr[:,i]))
  Tval=reduce(np.dot,(vr[:,i].T,T,vr[:,i]))
  print(i,Tval,Jval)

'''
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
