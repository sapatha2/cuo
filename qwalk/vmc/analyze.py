import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import statsmodels.api as sm 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit

df=None
for fin in ['run1s/ex2_s_Ndet10_gsw0.7_gosling.pickle','run2a/ex2_a_Ndet10_gsw0.8_gosling.pickle']:
  if(df is None): df=pd.read_pickle(fin)
  else: df=pd.concat((df,pd.read_pickle(fin)),axis=0)

#X=df[['energy','V_3d','V_3d_2p','V_4s_2p','V_4s_3d']]
#X=df[['energy','J_3d','J_3d_2p','J_4s_2p','J_4s_3d']]
#X=df[['energy','U_3d','U_3dd','U_3dpi','U_3dz2','U_2ppi','U_2pz','U_4s']]
X=df[['energy','n_3d','n_2p','n_4s','n_3dz2','n_3dpi','n_3dd','n_2ppi','n_2pz']]
#X=df[['energy','4s-2pz','4s-3dz2','3dz2-2pz','3dpi-2ppi']]

#sns.pairplot(X)
#plt.show()

y=df['energy']
X=df[['V_3d','V_4s_2p','V_3d_2p','V_4s_3d','J_4s_2p','J_4s_3d','n_3d','n_2p','n_4s','4s-2pz','4s-3dz2','3dz2-2pz','3dpi-2ppi']]
#OMP
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
for i in range(1,X.shape[1]+1):
#for i in range(41,42):
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
  print(omp.coef_[ind])
'''
  plt.title(fname)
  plt.xlabel("Predicted energy (eV)")
  plt.ylabel("DFT Energy (eV)")
  plt.plot(omp.predict(X),y,'og')
  plt.plot(y,y,'b-')
  plt.savefig(fname.split("p")[0][:-1]+".fit_fix.pdf",bbox_inches='tight')
  #plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-',label="Nparms= "+str(i))
  '''
'''
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(list(X))),list(X),rotation=90)
plt.title(fname)
plt.ylabel("Parameter (eV)")
plt.legend(loc='best')
plt.savefig(fname.split("p")[0][:-1]+".omp_fix.pdf",bbox_inches='tight')
'''
