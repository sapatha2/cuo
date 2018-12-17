import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
import statsmodels.api as sm 
#from sklearn.linear_model import OrthogonalMatchingPursuit
#from sklearn.model_selection import cross_val_score

#Load
detgen=['s']*10+['a']*10
N=['200']*20
Ndet=['2']+['5']*9+['2']+['5']*9
c=['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
c+=c

data=None
for i in range(len(detgen)):
  f='b3lyp_iao_b_'+detgen[i]+'_N'+N[i]+'_Ndet'+Ndet[i]+'_c'+c[i]+'.pickle'
  df=pd.read_pickle(f)

  #1-body Linear regression 
  y=df['E']
  X=df.drop(columns=['E'])
  ind=[]
  for z in list(X):
    if('_u' in z): X=X.drop(columns=[z])

  if(Ndet[i]=='2'):
    pass
  else:
    ols=sm.OLS(y,X).fit()
    
    params=ols.params
    errs=  ols.bse
    data_row=np.concatenate((params.values,errs.values))
    data_row=np.concatenate((np.array([detgen[i],N[i],Ndet[i],c[i]]),data_row))
    if(data is None): data=data_row
    else: 
      if(len(data.shape)<2): data=data[:,np.newaxis]
      data=np.concatenate((data,data_row[:,np.newaxis]),axis=1)

err_labels=errs.index
err_labels=[x+'_err' for x in err_labels]
labels=["detgen","N","Ndet","c"]+list(params.index)+list(err_labels)
data=pd.DataFrame(data.T,columns=labels)

#Plot parameter values
ex=['s','a']
c=['r','g']
marker=['v','s']
for i in range(2):
  detgen=ex[i]
  sub_df=data[data['detgen']==detgen].drop(columns=['detgen','N','Ndet','c'])
  j=0
  for p in list(sub_df):
    if('err' in p): pass
    else:
      y=np.array(sub_df[p].values,dtype=float)
      yerr=np.array(sub_df[p+'_err'].values,dtype=float)
      x=j+np.random.normal(size=len(y),scale=0.1)
      if(j==0):
        plt.errorbar(x,y,yerr=yerr,fmt=c[i]+marker[i],label='detgen='+detgen)
      else:
        plt.errorbar(x,y,yerr=yerr,fmt=c[i]+marker[i])
      j+=1

#ex_analysis1.pdf
plt.ylabel("Parameter value (eV)")
plt.xlabel("Parameter")
plt.xticks(np.arange(j+1),list(sub_df)[:10])
plt.axhline(0.0,color='k',ls='--')
plt.legend(loc='best')
plt.title("1-body fit, varying sampling schemes")
plt.show()
