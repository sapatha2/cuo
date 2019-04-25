import numpy as np
import scipy.linalg
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#PYSCF ordering: 2rdm[p,r,q,s]_x,x' = <cp_x^+ cq_x'^+ cs_x' cr_x>
h2=np.zeros((3,3,3,3,3))
index=[0,1,2]
#for p in range(len(index)):
#  for j in range(p+1,len(index)):
j=2
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

'''
h2[0][s,s,i,i]=0.25
h2[0][i,i,s,s]=0.25
h2[2][s,s,i,i]=0.25
h2[2][i,i,s,s]=0.25

h2[1][i,i,s,s]=-0.25
h2[1][s,s,i,i]=-0.25
h2[1][s,i,i,s]=-0.50
h2[1][i,s,s,i]=-0.50
'''

h1=np.zeros((3,3,3))
#FCI Broken (Based off of pyscf/fci/direct_uhf.py __main__)
mol = gto.Mole()
cis = fci.direct_uhf.FCISolver(mol)
eri_aa = ao2mo.restore(1, h2[0], 3)
eri_ab = ao2mo.restore(1, h2[1], 3)
eri_bb = ao2mo.restore(1, h2[2], 3) #4 and 8 fold identical since we have no complex entries
eri = (eri_aa, eri_ab, eri_bb)

nelec=(3,0)
nroots=100
norb=3
e, ci = fci.direct_uhf.kernel(h1, eri, norb, nelec, nroots=nroots)
print(e)

'''
sigJ = []
for i in range(nroots):
  dm2=cis.make_rdm12s(ci[i],norb,nelec)
  Jsd = 0
  for i in [1]:
    Jsd += 0.25*(dm2[1][0][0,0,i,i] + dm2[1][2][0,0,i,i] - dm2[1][1][0,0,i,i] - dm2[1][1][i,i,0,0])-\
           0.5*(dm2[1][1][i,0,0,i] + dm2[1][1][0,i,i,0])
  print(Jsd)
'''
