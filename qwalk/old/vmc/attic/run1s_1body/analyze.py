import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit

fname='s_Ndet10_gsw0.7_df.pickleR'
df=pd.read_pickle(fname)
#sns.pairplot(df.drop(columns=['energy_err']))
#plt.savefig('s_Ndet10_gsw0.7_pairplot.pdf')
#exit(0)

y=df['energy']
X=df.drop(columns=['energy','energy_err','obdm_3dd_3dd'])

#OLS
X=sm.add_constant(X)
ols=sm.OLS(y,X).fit()
print(ols.summary())

plt.ylabel('E_VMC (eV)')
plt.xlabel('E_Pred (eV)')
plt.errorbar(ols.predict(X),y,yerr=df['energy_err'],fmt='bo')
plt.plot(y,y,'g--')
plt.title('Full active space 1-body fit')
#plt.savefig('s_Ndet10_gsw0.7_df.pdf')
plt.show()

#OMP
'''
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
for i in range(1,X.shape[1]+1):
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

  plt.title(fname)
  plt.xlabel("Predicted energy (eV)")
  plt.ylabel("DFT Energy (eV)")
  plt.plot(omp.predict(X),y,'og')
  plt.plot(y,y,'b-')
  #plt.savefig(fname.split("p")[0][:-1]+".fit_fix.pdf",bbox_inches='tight')
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
