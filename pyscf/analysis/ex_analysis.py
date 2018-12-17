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

#Load

#Pairplot
#sns.pairplot(df)
#plt.savefig(f.split('.')[0]+'_pp.pdf',bbox_inches='tight')

#Matrix rank check
y=df['E']
X=df.drop(columns=['E'])
for z in list(X):
  if("u" in z):
    X=X.drop(columns=[z])
u,s,v=np.linalg.svd(X)
rank=np.linalg.matrix_rank(X,tol=1e-6)
print(s)
print('N parms: ', X.shape[1])
print('Rank data matrix: ',rank)

#Linear regression 
#X=sm.add_constant(X)
model=sm.OLS(y,X)
res_ols=model.fit()
print(res_ols.summary())
yhat=res_ols.predict(X)
plt.xlabel("Actual eV")
plt.plot(y,yhat,'bo')
plt.plot(y,y,'-')
plt.show()

#OMP
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
for i in range(1,X.shape[1]+1):
#for i in range(1,9):
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
  
  #Formatted print 
  names=np.array(list(np.array(list(X))[ind])+["E0"])
  vals=np.array(list(omp.coef_[ind])+[omp.intercept_])
  print(len(names),len(vals))
  for i in range(len(names)):
    print(str(names[i])+": "+str(vals[i]))
  #print(np.array(list(X))[ind])
  #print(omp.coef_[ind],omp.intercept_)
  
  plt.xlabel("Predicted energy (eV)")
  plt.ylabel("DFT Energy (eV)")
  plt.plot(omp.predict(X),y,'og')
  plt.plot(y,y,'b-')
  #plt.savefig(fname.split("p")[0][:-1]+".fit_fix.pdf",bbox_inches='tight')
  plt.show()
  #plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-',label="Nparms= "+str(i))
'''
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(list(X))),list(X),rotation=90)
plt.ylabel("Parameter (eV)")
plt.legend(loc='best')
#plt.savefig(fname.split("p")[0][:-1]+".omp_fix.pdf",bbox_inches='tight')
plt.show()
'''
