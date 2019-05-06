#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS,UKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from functools import reduce 
import seaborn as sns 

#MO basis 
'''
chkfile='../chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
mol=lib.chkfile.load_mol(chkfile)
m=ROKS(mol)
m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
a=m.mo_coeff[:,:14]
'''

#IAO basis
a = pd.read_pickle('../ub3lyp_full/b3lyp_iao_b.pickle')

charge=0
S=1
r=1.725
method='UB3LYP'
basis='vtz'
el='Cu'

chkfiles = ['../ub3lyp_full/Cuvtz_r1.725_s1_UB3LYP_'+str(i)+'.chk' for i in range(11)]
chkfiles += ['../ub3lyp_full/Cuvtz_r1.725_s3_UB3LYP_'+str(i)+'.chk' for i in range(6)]
chkfiles += ['Cuvtz_r1.725_s1_UB3LYP_11.chk',
'Cuvtz_r1.725_s1_UB3LYP_12.chk',
'Cuvtz_r1.725_s3_UB3LYP_13.chk']

zz=0
ns=[]
nd=[]
npi=[]
nz=[]
tpi=[]
tsz=[]
tdz=[]
tds=[]
for chkfile in chkfiles:
  mol=lib.chkfile.load_mol(chkfile)
  m=UKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

  #Build RDM on IAO basis 
  s=m.get_ovlp()
  mo_occ=m.mo_occ
  M=m.mo_coeff[0][:,mo_occ[0]>0]
  M=reduce(np.dot,(a.T,s,M))
  dm_u=np.dot(M,M.T)
  M=m.mo_coeff[1][:,mo_occ[1]>0]
  M=reduce(np.dot,(a.T,s,M))
  dm_d=np.dot(M,M.T)

  #MOs
  #labels=['dx','dy','dz','dd','dd','px','py','pz','4s']
  #dm_u = dm_u[5:][:,5:]
  #dm_d = dm_d[5:][:,5:]

  #IAOs
  labels=['4s','dd','dy','dz','dx','dd','px','py','pz']
  ind=[1,5,6,7,8,9,11,12,13]
  dm_u = dm_u[ind][:,ind]
  dm_d = dm_d[ind][:,ind]
  dm = dm_u + dm_d
  print(np.trace(dm))

  ns.append( dm[0,0])
  nd.append( np.sum(dm[[1,2,3,4,5],[1,2,3,4,5]]))
  npi.append( dm[6,6]+dm[7,7])
  nz.append( dm[8,8])
  tpi.append( 2*(dm[2,7] + dm[4,6]))
  tsz.append( 2*dm[0,8])
  tdz.append( 2*dm[3,8])
  tds.append( 2*dm[0,3])
  
df = pd.DataFrame({'ns':ns,'nd':nd,'npi':npi,'nz':nz,'tpi':tpi,'tsz':tsz,'tdz':tdz,'tds':tds,'ind':np.arange(len(tds))})
#sns.pairplot(df,vars=['tpi','tsz','tds','tdz'],hue='ind',markers=['.']*17+['o']*3)
sns.pairplot(df,vars=['nd','nz','npi','ns'],hue='ind',markers=['.']*17+['o']*3)
plt.show()
