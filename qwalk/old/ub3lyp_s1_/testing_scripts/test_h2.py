import numpy as np
import scipy.linalg
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

norb = 3
nelec=(3,0)
nroots=9

#PYSCF ordering: 2rdm[p,r,q,s]_x,x' = <cp_x^+ cq_x'^+ cs_x' cr_x>
h2=np.zeros((3,norb,norb,norb,norb))
index=np.arange(norb)
#for p in range(len(index)):
#  for j in range(p+1,len(index)):
j = norb - 1
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

h1=np.zeros((norb,norb,norb))
#FCI Broken (Based off of pyscf/fci/direct_uhf.py __main__)
mol = gto.Mole()
cis = fci.direct_uhf.FCISolver(mol)
eri_aa = ao2mo.restore(1, h2[0], norb)
eri_ab = ao2mo.restore(1, h2[1], norb)
eri_bb = ao2mo.restore(1, h2[2], norb) #4 and 8 fold identical since we have no complex entries
eri = (eri_aa, eri_ab, eri_bb)

e, ci = fci.direct_uhf.kernel(h1, eri, norb, nelec, nroots=nroots)
print(e)

sigJ = []
for i in range(len(ci)):
  dm2=cis.make_rdm12s(ci[i],norb,nelec)
  Jsd = 0
  for i in range(len(index)-1):
    Jsd += 0.25*(dm2[1][0][j,j,i,i] + dm2[1][2][j,j,i,i] - dm2[1][1][j,j,i,i] - dm2[1][1][i,i,j,j])-\
           0.5*(dm2[1][1][i,j,j,i] + dm2[1][1][j,i,i,j])
  print(Jsd)
