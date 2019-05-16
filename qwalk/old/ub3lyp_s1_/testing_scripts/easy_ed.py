#Exact diagonalization routine in pyscf
import numpy as np
import scipy.linalg
from pyscf import gto, scf, ao2mo, cc, fci, mcscf

def hubbard(t,U):
  H=np.array([[0,0,-t,-t],[0,0,t,t],[-t,t,U,0],[-t,t,0,U]])
  w,vr=np.linalg.eigh(H)
  print(w)

#FCI Broken (Based off of pyscf/fci/direct_uhf.py __main__)
#Build Hamiltonian objects
t, u, n = 1, 4, 2
nea, neb = 1, 1 

h1 = np.zeros((2,n,n))     #a, b
for i in range(n-1):
   h1[:,i,i+1] = h1[:,i+1,i] = t
h2 = np.zeros((3,n,n,n,n)) #aa, ab, bb

#Build mol object
mol = gto.Mole()

#CASCI
cis = fci.direct_uhf.FCISolver(mol)
norb = n 
nelec = (nea, neb)

h2[0]=0
h2[2]=0
h2[1][0,0,0,0]=u
h2[1][1,1,1,1]=u
eri_aa = ao2mo.restore(1, h2[0], n)
eri_ab = ao2mo.restore(1, h2[1], n)
eri_bb = ao2mo.restore(1, h2[2], n)
eri = (eri_aa, eri_ab, eri_bb)

e, ci = fci.direct_uhf.kernel(h1, eri, norb, nelec, nroots=4)
print(e)
for i in range(len(ci)): print(ci[i])
hubbard(t,u)

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
nelecas=(1,1) 
casscf=mcscf.CASCI(mf,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
casscf.kernel()
casscf.verbose=4
hubbard(t,u)
'''

#FCI broken
'''
from functools import reduce
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import cistring

mol = gto.Mole()
mol.verbose = 0
mol.output = None#"out_h2o"
mol.atom = [
    ['H', ( 1.,-1.    , 0.   )],
    ['H', ( 0.,-1.    ,-1.   )],
    ['H', ( 1.,-0.5   ,-1.   )],
    #['H', ( 0.,-0.5   ,-1.   )],
    #['H', ( 0.,-0.5   ,-0.   )],
    ['H', ( 0.,-0.    ,-1.   )],
    ['H', ( 1.,-0.5   , 0.   )],
    ['H', ( 0., 1.    , 1.   )],
]

mol.basis = {'H': 'sto-3g'}
mol.charge = 1
mol.spin = 1
mol.build()

m = scf.UHF(mol)
ehf = m.scf()

cis = fci.direct_uhf.FCISolver(mol)
norb = m.mo_energy[0].size
nea = (mol.nelectron+1) // 2
neb = (mol.nelectron-1) // 2
nelec = (nea, neb)
mo_a = m.mo_coeff[0]
mo_b = m.mo_coeff[1]
h1e_a = reduce(np.dot, (mo_a.T, m.get_hcore(), mo_a))
h1e_b = reduce(np.dot, (mo_b.T, m.get_hcore(), mo_b))
g2e_aa = ao2mo.incore.general(m._eri, (mo_a,)*4, compact=False)
g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
g2e_ab = ao2mo.incore.general(m._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
g2e_bb = ao2mo.incore.general(m._eri, (mo_b,)*4, compact=False)
g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
h1e = (h1e_a, h1e_b)
eri = (g2e_aa, g2e_ab, g2e_bb)
na = cistring.num_strings(norb, nea)
nb = cistring.num_strings(norb, neb)
np.random.seed(15)
fcivec = np.random.random((na,nb))

e = fci.direct_uhf.kernel(h1e, eri, norb, nelec)[0]
print(e, e--8.65159903476)
'''
