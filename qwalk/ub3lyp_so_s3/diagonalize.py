import numpy as np 
import matplotlib.pyplot as plt
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
