#Exact diagonalization routine in pyscf
import numpy as np
import scipy.linalg
from pyscf import gto, scf, ao2mo, cc, fci, mcscf

'''
mol = gto.M(atom='H 0 0 0; H 0 0 1.2', basis='ccpvdz')
mf = scf.UHF(mol)
mf.kernel()
print(mf.get_hcore().shape)
print(mf.get_ovlp().shape)
print(mf.mo_occ.shape)
print(mf.get_veff(mol,mf.make_rdm1()).shape)
exit(0)
'''

def hubbard(t,U):
  H=np.array([[0,0,-t,-t],[0,0,t,t],[-t,t,U,0],[-t,t,0,U]])
  w,vr=np.linalg.eigh(H)
  print(w)

#CCSD custom Hamiltonian
'''
mol = gto.Mole(verbose=4)
mol.nelectron = n = 2
t, u = 1., 4.
mf = scf.RHF(mol)
h1 = np.zeros((n,n))
for i in range(n-1):
   h1[i,i+1] = h1[i+1,i] = t
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(n)
h2 = np.zeros((n,n,n,n))
for i in range(n):
  h2[i,i,i,i] = u

# 2e Hamiltonian in 4-fold symmetry
mf._eri = ao2mo.restore(4, h2, n)
mf.run()

cc = cc.UCCSD(mf)
cc.kernel()
e_ip, c_ip = cc.ipccsd(nroots=3)
print(e_ip)

hubbard(t,u)
'''

#FCI
'''
mol = gto.Mole(verbose=4)
mol.nelectron = n = 2
t, u = 1., 4.
mf = scf.RHF(mol)
h1 = np.zeros((n,n))
for i in range(n-1):
   h1[i,i+1] = h1[i+1,i] = t
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(n)
h2 = np.zeros((n,n,n,n))
for i in range(n):
  h2[i,i,i,i] = u

# 2e Hamiltonian in 4-fold symmetry
mf._eri = ao2mo.restore(4, h2, n)
mf.run()

state_id=0 #which state to calculate, 0 is GS
ncore=0
ncas=2
nelecas=(2,0) 
casscf=mcscf.CASCI(mf,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
casscf.kernel()
casscf.verbose=4
hubbard(t,u)
'''

#HF, 
mol = gto.Mole(verbose=4)
mol.nelectron = n = 2
t, U = 1, 4
mf = scf.UHF(mol)

h1=np.zeros((2,n,n))
for i in range(n-1):
   h1[:,i,i+1] = h1[:,i+1,i] = t
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(n)

h2 = np.zeros((2,n,n,n,n))
'''
#h2[p,r,q,s] = <p+ q+ s r> 
h2[0,0,1,1]=J
h2[1,1,0,0]=J
'''

# 2e Hamiltonian in 4-fold symmetry
mf._eri = ao2mo.restore(4, h2, n)
dm0=np.array(
[[[1,0],[0,1]]],
[[[0,0],[0,0]]]
)

mf.run(dm0)
exit(0)

state_id=0 #which state to calculate, 0 is GS
ncore=0
ncas=2
nelecas=(1,1) 
casscf=mcscf.CASCI(mf,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
casscf.verbose=4
casscf.kernel()
print(casscf.ci)
