import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
import statsmodels.api as sm 
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
from functools import reduce

df=pd.read_pickle('b3lyp_mo_symm_a_N100_Ndet50_c0.5_beta0.pickle')
y=df['E']
X=df.drop(columns=['E'])

#Traces check
'''
tr=X.sum(axis=1)
plt.plot(tr,'o')
plt.show()
'''
#Linear regression
ols=sm.OLS(y,X).fit()
#print(ols.summary())
#plt.plot(ols.predict(X),y,'o')
#plt.show()

#Rotate to IAO basis
opt_mo=pd.read_pickle('b3lyp_mo_symm.pickle')
iao=pd.read_pickle('../analysis/b3lyp_iao_b.pickle')
chkfile='../chkfiles/Cuvtz_r1.963925_c0_s-1_B3LYP.chk'
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile,'scf'))
s=m.get_ovlp()

e1_mo=np.diag(ols.params)
M=reduce(np.dot,(opt_mo.T,s,iao))
e1_iao=reduce(np.dot,(M.T,e1_mo,M))

e1_base=np.diag(m.mo_energy)
M=reduce(np.dot,(opt_mo.T,s,iao))
e1_iao_base=reduce(np.dot,(M.T,e1_mo,M))

plt.matshow(e1_iao_base-e1_iao,cmap=plt.cm.bwr,vmin=-2,vmax=2)
plt.show()
