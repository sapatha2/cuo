#Testing MO optimization 
from scipy.optimize import minimize
import numpy as np 
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
from functools import reduce 

#Consider a set of vectors, want to find the closest vector to them all (L2 distance)
def optmo_test(vs,v0):
  '''
  input
  vs - shape: nvectors x ncomps
  v0 - guess for v_opt solution
  output
  v_opt - ncomps
  '''

  #Cost function
  def f(x):
    cost=0
    for i in range(vs.shape[0]): 
      cost+=np.dot(vs[i]-x,vs[i]-x)
    return cost
  
  #Constrained optimization 
  const=({'type':'eq', 'fun':lambda x: np.dot(x,x)-1})
  v_opt=minimize(f,v0,constraints=const) 
  return v_opt

#Consider a set of vectors, want to find the closest vector to them all (L2 distance)
def optmo(vs,v0,ovlp,symm=False,n=None):
  '''
  input
  vs - shape: nvectors x ncomps
  v0 - guess for v_opt solution, solution will satisfy this symmetry
  symm - if you have multiple orbitals in the symmetry group to optimize 

  output
  v_opt - ncomps
  '''

  #Fix signs relative to v0
  signind=np.where(v0!=0)[0][0]
  for i in range(vs.shape[0]):
    vs[i,:]*=np.sign(vs[i,signind]*v0[signind])

  #Cost function
  def f(x):
    cost=0
    #Symmetry 
    x[np.where(v0==0)[0]]=0
    for i in range(vs.shape[0]): 
      cost+=np.dot(vs[i]-x,vs[i]-x)
    return cost
  
  if(symm): 
    assert(n is not None)
    const=()
    N=int(vs.shape[1]/n)
    for i in range(n):
      #Normality
      const+=({'type':'eq', 'fun':lambda x: reduce(np.dot,(x[N*i:N*(i+1)],ovlp,x[N*i:N*(i+1)]))-1},)
    for i in range(n-1):
      const+=({'type':'eq', 'fun':lambda x: reduce(np.dot,(x[N*i:N*(i+1)],ovlp,x[N*(i+1):N*(i+2)]))},)
  else:
    #Normality
    const=({'type':'eq', 'fun':lambda x: reduce(np.dot,(x,ovlp,x))-1},)
  
  v_opt=minimize(f,v0,constraints=const)
  return v_opt

#Gather vectors we want to use
def gathermo_vecs(flist,act_orbs):
  '''
  input
  flist - chkfile list 
  act_orbs - orbitals you want to optimize basis for
  output
  vecs_list - vectors list, shape: act_mo x nvectors x ncomps
  ovlp - overlap necessary for constraints
  '''
  vecs_list=[]
  for f,a in zip(flist,act_orbs):
    mol=lib.chkfile.load_mol(f)
    m=ROHF(mol)
    m.__dict__.update(lib.chkfile.load(f, 'scf'))
    vecs=m.mo_coeff[:,a].T
    vecs_list.append(vecs)
  vecs_list=np.array(vecs_list)
  vecs_list=np.einsum('ijk->jik',vecs_list)
  return vecs_list,m.get_ovlp()

if __name__=='__main__':
  #optmo_test
  #vs=np.array([[0,1],[1,0]])
  #v0=np.array([0.6,0.4])
  #v_opt=optmo_test(vs,v0)
  #print(v_opt)

  #Gather active MOs
  flist=['../chkfiles/Cuvtz_r1.963925_c0_s-1_B3LYP.chk',
  '../chkfiles/Cuvtz_r1.963925_c0_s1_B3LYP.chk',
  '../full_chk/Cuvtz_r1.963925_c0_s1_B3LYP_2Y.chk',
  '../chkfiles/Cuvtz_r1.963925_c0_s3_B3LYP.chk']
  #Order:    dz2,yz,xz,z,y,x,s
  act_orbs=[[5,6,7,10,11,12,13],
  np.array([5,7,6,10,12,11,13]),
  np.array([7,6,5,12,11,10,13]),
  np.array([5,7,6,10,12,11,13])]
  vecs_list,ovlp=gathermo_vecs(flist,act_orbs)
  
  #Optimize MOs
  v_opt_list=[]
  
  #pi orbitals
  n=2
  vs=np.concatenate((vecs_list[1,:,:],vecs_list[4,:,:]),axis=1)
  v_opt=optmo(vs,vs[2],ovlp,symm=True,n=n)
  print(v_opt.fun)
  N=int(vs.shape[1]/2)
  v_opt_list.append(v_opt.x[:N])
  v_opt_list.append(v_opt.x[N:])
  
  vs=np.concatenate((vecs_list[2,:,:],vecs_list[5,:,:]),axis=1)
  v_opt=optmo(vs,vs[2],ovlp,symm=True,n=n)
  print(v_opt.fun)
  N=int(vs.shape[1]/2)
  v_opt_list.append(v_opt.x[:N])
  v_opt_list.append(v_opt.x[N:])
  
  #sigma orbitals
  n=3
  vs=np.concatenate((vecs_list[0,:,:],vecs_list[3,:,:],vecs_list[6,:,:]),axis=1)
  v_opt=optmo(vs,vs[3],ovlp,symm=True,n=n)
  print(v_opt.fun)
  N=int(vs.shape[1]/3)
  v_opt_list.append(v_opt.x[:N])
  v_opt_list.append(v_opt.x[N:2*N])
  v_opt_list.append(v_opt.x[2*N:])
  
  v_opt_list=np.array(v_opt_list).T
  print(v_opt_list.shape)

  #check overlaps
  for i in range(v_opt_list.shape[1]):
    for j in range(i+1,v_opt_list.shape[1]):
      s=reduce(np.dot,(v_opt_list[:,i],ovlp,v_opt_list[:,j]))
      print(i,j,s)

  #Write to qwalk plot
  from pyscf2qwalk import print_qwalk_mol
  mol=lib.chkfile.load_mol(flist[0])
  m=ROHF(mol)
  m.__dict__.update(lib.chkfile.load(flist[0], 'scf'))
  m.mo_coeff[:,act_orbs[0]]=v_opt_list
  print_qwalk_mol(mol,m,basename='./plots/qw')

  #Write to pickle
  v_opt_list.dump('b3lyp_mo_symm.pickle')
