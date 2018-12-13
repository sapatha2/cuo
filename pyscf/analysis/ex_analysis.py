import numpy as np 
import matplotlib.pyplot as plt
from methods import gensingles,genex
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
from functools import reduce
import pandas as pd 
import seaborn as sns 
import statsmodels.api as sm 
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.model_selection import cross_val_score

f='b3lyp_iao_b.pickle'
a=np.load(f)

r=1.963925
method='B3LYP'
basis='vtz'
el='Cu'
charge=0

escf={'-1':-213.5799904,
       '1':-213.5799904,
       '3':-213.522801}
occ={'-1': [np.arange(6,14)-1,np.arange(6,13)-1], 
      '1': [np.arange(6,14)-1,np.arange(6,13)-1],
      '3': [np.arange(6,15)-1,np.arange(6,12)-1]}
virt={'-1': [np.arange(14,15)-1,np.arange(13,15)-1], 
       '1': [np.arange(14,15)-1,np.arange(13,15)-1], 
       '3': [np.arange(15,15)-1,np.arange(12,15)-1]}
ncore={'-1':5,
        '1':5,
        '3':5}
nact={'-1':[8,7],
       '1':[8,7],
       '3':[9,6]}
act={'-1':[np.arange(5,15),np.arange(5,15)],
      '1':[np.arange(5,15),np.arange(5,15)],
      '3':[np.arange(5,15),np.arange(5,15)]}

data=None
for mol_spin in [-1,1,3]:
  chkfile="../chkfiles/"+el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method+".chk"
  mol=lib.chkfile.load_mol(chkfile)

  if("U" in method): m=UHF(mol)
  else: m=ROHF(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

  ##################
  #BUILD EXCITATIONS 
  mo_occ=np.array([np.ceil(m.mo_occ-m.mo_occ/2),np.floor(m.mo_occ/2)])
  
  #Singles excitations only 
  #dm_list,__,__,__=gensingles(mo_occ,occ[str(mol_spin)],virt[str(mol_spin)])
  
  #Arbitrary excitations
  detgen='s'
  N=100
  Ndet=2
  c=0.0
  st=str(mol_spin)
  dm_list=genex(mo_occ,a,ncore[st],act[st],nact[st],N,Ndet,detgen,c)

  #IAO rdms
  s=m.get_ovlp()
  M=m.mo_coeff
  M=reduce(np.dot,(a.T,s,M))
  iao_dm_list=np.einsum('ijkl,mk->ijml',dm_list,M)
  iao_dm_list=np.einsum('ijml,nl->ijmn',iao_dm_list,M)
  ##################

  #Traces
  '''
  plt.subplot(211)
  plt.title("MO trace singles, S="+str(mol_spin)+", IAO="+f)
  tr_mo=np.einsum('ijmm->ij',dm_list)
  plt.plot(tr_mo[:,0],'*')
  plt.plot(tr_mo[:,1],'s')
  
  plt.subplot(212)
  plt.title("IAO trace singles, S="+str(mol_spin)+", IAO="+f)
  tr=np.einsum('ijmm->ij',iao_dm_list)
  plt.plot(tr[:,0],'*')
  plt.plot(tr[:,1],'s')
 
  plt.savefig(f.split(".")[0]+'_s'+str(mol_spin)+'_tr.pdf',bbox_inches='tight')
  plt.close()
  '''

  #Energies
  e=np.einsum('ijkl,l->ijk',dm_list,m.mo_energy)
  e=np.einsum('ijk->ij',e)
  e=e[:,0]+e[:,1]
  e-=e[0] #Eigenvalue difference
  e+=escf[str(mol_spin)] #Base SCF difference
  
  #Number occupations 
  n=np.einsum('ijmm->ijm',iao_dm_list)
  labels=np.array(["3s","4s","3px","3py","3pz","3dxy","3dyz","3dz2","3dxz","3dx2y2","2s","2px","2py","2pz"])
  rel=[1,5,6,7,8,9,11,12,13]
  n=n[:,0,rel]+n[:,1,rel]
  
  #Hopping 
  trel=np.array([[1,1,6,8,7],[7,13,12,11,13]])
  tlabels=np.array([labels[trel[0]],labels[trel[1]]]).T
  tlabels=[x[0]+"-"+x[1] for x in tlabels]
  t=iao_dm_list[:,0,trel[0],trel[1]]+iao_dm_list[:,1,trel[0],trel[1]]

  #Data object
  d=np.concatenate((e[:,np.newaxis],n,t),axis=1)
  if(data is None): data=d
  else: data=np.concatenate((data,d),axis=0)

#Full data frame
df=pd.DataFrame(data,columns=["E"]+list(labels[rel])+list(tlabels))
df['E']-=df['E'][0]
df['E']*=27.2114
df['3dd']=df['3dxy']+df['3dx2y2']
df['3dpi']=df['3dxz']+df['3dyz']
df['2ppi']=df['2px']+df['2py']
df['tpi']=df['3dyz-2py']+df['3dxz-2px']
df=df.drop(columns=['3dxz','3dyz','2px','2py','3dyz-2py','3dxz-2px','3dx2y2','3dxy'])

#Pairplot
#sns.pairplot(df)
#plt.savefig(f.split('.')[0]+'_pp.pdf',bbox_inches='tight')

#Matrix rank check
y=df['E']
X=df.drop(columns=['E'])
u,s,v=np.linalg.svd(X)
rank=np.linalg.matrix_rank(X,tol=1e-6)
print(s)
print('N parms: ', X.shape[1])
print('Rank data matrix: ',rank)

#Linear regression 
'''
X=df['4s']
model=sm.OLS(y,X)
res_ols=model.fit()
print(res_ols.summary())
yhat=res_ols.predict(X)
plt.plot(y,yhat,'bo')
plt.plot(y,y,'-')
plt.show()
'''

#OMP
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
#for i in range(1,X.shape[1]+1):
for i in range(1,9):
  print("n_nonzero_coefs="+str(i))
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i)
  omp.fit(X,y)
  nparms.append(i)
  scores.append(omp.score(X,y))
  tmp=cross_val_score(omp,X,y,cv=5)
  cscores.append(tmp.mean())
  cscores_err.append(tmp.std()*2)
  print("R2: ",omp.score(X,y))
  print("R2CV: ",tmp.mean(),"(",tmp.std()*2,")")
  ind=np.abs(omp.coef_)>0
  Xr=X.values[:,ind]
  conds.append(np.linalg.cond(Xr))
  print("Cond: ",np.linalg.cond(Xr))
  print(np.array(list(X))[ind])
  print(omp.coef_[ind],omp.intercept_)
  
  #plt.xlabel("Predicted energy (eV)")
  #plt.ylabel("DFT Energy (eV)")
  #plt.plot(omp.predict(X),y,'og')
  #plt.plot(y,y,'b-')
  #plt.savefig(fname.split("p")[0][:-1]+".fit_fix.pdf",bbox_inches='tight')
  #plt.show()
  plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-',label="Nparms= "+str(i))
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(list(X))),list(X),rotation=90)
plt.ylabel("Parameter (eV)")
plt.legend(loc='best')
#plt.savefig(fname.split("p")[0][:-1]+".omp_fix.pdf",bbox_inches='tight')
plt.show()
