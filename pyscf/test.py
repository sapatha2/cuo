from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF
import numpy as np 
from functools import reduce
from scipy.misc import comb 
import pandas as pd
import statsmodels.api as sm 
import matplotlib.pyplot as plt 

#Read in SCF file
#chkfile="chkfiles/Cuvtz_r1.963925_c0_s1_B3LYP.chk"
chkfile="chkfiles/Cuvtz_r1.963925_c0_s3_B3LYP.chk"
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
s=m.get_ovlp()
M=np.dot(s,m.mo_coeff) #AO -> MO 
a=np.load('analysis/b3lyp_iao_b.pickle') 
M2=np.dot(s,a) #AO -> IAO

#Full active space
ncore=5
#nelecas=(8,7)
nelecas=(9,6)
ncas=9
mc=mcscf.CASCI(m,ncore=ncore,ncas=ncas,nelecas=nelecas)

#USER INPUT
N=100
Ndet=10
gsw=0.8

e_list=[]
dm_list=[]
for i in range(N+1):
  #Build weight vector
  ci=np.zeros(comb(ncas,nelecas[0],exact=True)*comb(ncas,nelecas[1],exact=True))
  if(i>0):
    gauss=np.random.normal(size=Ndet-1)
    gauss/=np.sqrt(np.dot(gauss,gauss))
    ci[0]=np.sqrt(gsw)
    ci[np.random.choice(np.arange(1,len(ci)),size=Ndet-1,replace=False)]=\
    gauss*np.sqrt(1-gsw)
  else: ci[0]=1.

  dm1,dm2=mc.make_rdm1s(ci=ci) #Get DM on AO basis

  #Get DM on MO basis
  mo_dm1=reduce(np.dot,(M.T,dm1,M)) #MO basis check 
  mo_dm2=reduce(np.dot,(M.T,dm2,M)) #MO basis check 
 
  #Get energies
  e=np.trace(np.dot(mo_dm1+mo_dm2,np.diag(mc.mo_energy)))

  #Get DM on IAO basis
  iao_dm1=reduce(np.dot,(M2.T,dm1,M2)) 
  iao_dm2=reduce(np.dot,(M2.T,dm2,M2)) 

  e_list.append(e)
  dm_list.append([iao_dm1,iao_dm2])

e_list=np.array(e_list)
dm_list=np.array(dm_list)

#Do our typical reduction stuff
#Number occupations 
n=np.einsum('ijmm->ijm',dm_list)
labels=np.array(["3s","4s","3px","3py","3pz","3dxy","3dyz","3dz2","3dxz","3dx2y2","2s","2px","2py","2pz"])
rel=[1,5,6,7,8,9,11,12,13]
n=n[:,0,rel]+n[:,1,rel]

#Hopping 
trel=np.array([[1,1,6,8,7],[7,13,12,11,13]])
tlabels=np.array([labels[trel[0]],labels[trel[1]]]).T
tlabels=[x[0]+"-"+x[1] for x in tlabels]
t=dm_list[:,0,trel[0],trel[1]]+dm_list[:,0,trel[1],trel[0]]+\
  dm_list[:,1,trel[0],trel[1]]+dm_list[:,1,trel[1],trel[0]] #Hermitian conjugates required

#Dataframe
data=np.concatenate((e_list[:,np.newaxis],n,t),axis=1)
df=pd.DataFrame(data,columns=["E"]+list(labels[rel])+list(tlabels))
df['E']-=df['E'][0]
df['E']*=27.2114
df['3dd']=df['3dxy']+df['3dx2y2']
df['3dpi']=df['3dxz']+df['3dyz']
df['2ppi']=df['2px']+df['2py']
df['tpi']=df['3dyz-2py']+df['3dxz-2px']
df['3dz2-2pz']*=-1  #Get the sign structure correct
df['4s-2pz']*=-1    #Get the sign structure correct
df=df.drop(columns=['3dxz','3dyz','2px','2py','3dyz-2py','3dxz-2px','3dx2y2','3dxy'])

df.to_pickle('test1b.pickle')

'''
#OLS
y=df['E']
X=df.drop(columns=['E'])
ols=sm.OLS(y,X).fit()
print(ols.summary())
plt.plot(y,ols.predict(X),'o')
plt.plot(y,y)
plt.show()
'''
