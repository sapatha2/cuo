import numpy as np 
from pyscf import lib, gto, scf, ao2mo, cc, fci, mcscf
from pyscf.scf import ROKS
from functools import reduce 
import matplotlib.pyplot as plt 

#1-body 
#MO to IAO Matrix in the correct ordering 
#dxy, dy, dz, z, py, s
act_iao=[5,6,7,13,12,1]
iao=np.load('../../../../pyscf/ub3lyp_full/b3lyp_iao_b.pickle')
iao=iao[:,act_iao]

#LOAD IN MOS
#dxy, dy, dz, z, py, s
act_mo=[8,6,7,12,11,13]
chkfile='../../../../pyscf/chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
mol=lib.chkfile.load_mol(chkfile)
m=ROKS(mol)
m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
mo=m.mo_coeff[:,act_mo]
s=m.get_ovlp()
mo_to_iao = reduce(np.dot,(mo.T,s,iao))

#dd dpi dz z pi s tpi tdz tsz tds
#model 5
#MO =  [0,0,0,0.42,1.82,2.84,0,0,0,0]
#IAO = [0,0.18,0.26,0.82,1.63,2.10,-0.55,0.46,0.96,0.5]

#model 9
#MO =  [0,0,0,0.43,1.83,2.90,0.14,0,0,0]
#IAO = [0,0.10,0.28,0.87,1.75,2.16,-0.44,0.49,0.95,0.49]

#model 12 
#MO = [0,0,0,0.32,1.75,2.80,0,-0.27,0,0]
#IAO = [0,0.17,0.45,0.54,1.55,1.99,-0.52,0.51,1.05,0.38]

#model 21
#MO = [0,0,0,0.49,1.81,2.87,0.17,0,0,0.26]
#IAO = [0,0.1,0.39,0.74,1.72,2.16,-0.43,0.56,0.83,0.67]

#model 20
MO = [0,0,0,0.45,2.08,2.87,0.22,0,-0.9,0]
IAO = [0,0.08,0.48,1.46,2.00,1.3,-0.46,0.84,1.25,0.71]

#model 24 
#MO = [0,0,0,0.32,1.74,2.78,0,-0.21,0,0.11]
#IAO = [0,0.18,0.44,0.56,1.57,2.02,-0.52,0.53,0.98,0.49]

H1_MO = np.diag(MO[:6])
H1_IAO = np.diag(IAO[:6])

H1_MO[[1,4],[4,1]] = MO[6]
H1_MO[[2,3],[3,2]] = MO[7]
H1_MO[[3,5],[5,3]] = MO[8]
H1_MO[[2,5],[5,2]] = MO[9]

H1_IAO[[1,4],[4,1]] = IAO[6]
H1_IAO[[2,3],[3,2]] = IAO[7]
H1_IAO[[3,5],[5,3]] = IAO[8]
H1_IAO[[2,5],[5,2]] = IAO[9]

w,vr = np.linalg.eigh(H1_MO)
w2,vr2 = np.linalg.eigh(H1_IAO)

print(np.around(w,2))
print(np.around(w2,2))

ovlp = reduce(np.dot,(vr.T,mo_to_iao,vr2))
plt.matshow(ovlp,vmin=-1,vmax=1,cmap = plt.cm.bwr)
plt.show()

'''
#2-body
#Jsi = 0.25*(ns_u - ns_d)*(ni_u - ni_d) + 
#0.5*(cs_u^+ cs_d ci_d^+ ci_u + cs_d^+ cs_u ci_u^+ ci_d)
#PYSCF ordering: 2rdm[p,r,q,s]_x,x' = <cp_x^+ cq_x'^+ cs_x' cr_x>
norb = 3
nelec = (2,1)

h2=np.zeros((3,norb,norb,norb,norb))
s = 0 
for i in range(1,norb):
  h2[0][s,s,i,i]=0.25
  h2[0][i,i,s,s]=0.25 #
  h2[2][s,s,i,i]=0.25
  h2[2][i,i,s,s]=0.25 #Comment out these, the 2 electron triplet state is wrong!
                      #And all of 3 electron is wrong

  h2[1][i,i,s,s]=-0.25
  h2[1][s,s,i,i]=-0.25
  h2[1][s,i,i,s]=-0.50
  h2[1][i,s,s,i]=-0.50
h1 = np.zeros((2,norb,norb))

mol = gto.Mole()
cis = fci.direct_uhf.FCISolver(mol)
eri_aa = ao2mo.restore(1, h2[0], norb)
eri_ab = ao2mo.restore(1, h2[1], norb)
eri_bb = ao2mo.restore(1, h2[2], norb) #4 and 8 fold identical since we have no complex entries
eri = (eri_aa, eri_ab, eri_bb)

e, ci = fci.direct_uhf.kernel(h1, eri, norb, nelec, nroots=10)
print(e)
'''
