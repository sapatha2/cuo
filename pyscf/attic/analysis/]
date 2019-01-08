import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
import statsmodels.api as sm 
#from sklearn.linear_model import OrthogonalMatchingPursuit
#from sklearn.model_selection import cross_val_score

#Load
'''
detgen=['s']*5+['a']*5
N=['100']*10
Ndet=['10']*10
c=['0.9','0.8','0.7','0.6','0.5']
c+=c
'''
detgen=['a']
N=['100']
Ndet=['10']
c=['0.8']

data=None
for i in range(len(detgen)):
  f='b3lyp_iao_b_'+detgen[i]+'_N'+N[i]+'_Ndet'+Ndet[i]+'_c'+c[i]+'WITHD.pickle'
  #f='b3lyp_iao_test_'+detgen[i]+'_N'+N[i]+'_Ndet'+Ndet[i]+'_c'+c[i]+'TESTIAO.pickle'
  df=pd.read_pickle(f)

  #df=df[df['spin']==1.5]
  df['3d']=df['3dd']+df['3dpi']+df['3dz2']
  '''
  df['2p']=df['2ppi']+df['2pz']
  sns.pairplot(df,vars=['E','4s','3d','2p','2ppi','2pz','4s-2pz','3dz2-2pz','4s-3dz2','tpi'],hue='spin')
  plt.show()
  exit(0)
  '''
  #df['tr']=df['4s']+df['2ppi']+df['2pz']+df['3dz2']+df['3dpi']+df['3dd']
  #sns.scatterplot(x=np.arange(df.shape[0]),y='tr',data=df,hue='spin')
  #plt.show()
  #exit(0)

  #1-body Linear regression 
  y=df['E']
  #X=df[['4s','2pz','2ppi','4s-2pz','3dz2-2pz','4s-3dz2','tpi']]
  X=df['3d']
  X=sm.add_constant(X)
  #X=df.drop(columns=['E','spin','3dd','3dpi','3dz2'])
  #X=df.drop(columns=['E','3dd'])
  ind=[]
  for z in list(X):
    if('_u' in z): X=X.drop(columns=[z])
    if('nmo' in z): X=X.drop(columns=[z])

  beta=-3
  ols=sm.WLS(y,X,weights=np.exp(beta*(y-min(y)))).fit()
  #ols=sm.OLS(y,X).fit()
  print(ols.summary())
  plt.title('Ndet='+str(Ndet[i])+',c='+str(c[i]))

  #df=pd.read_pickle(f)
  #y=df['E']
  #X=df[['4s','2pz','4s-2pz']]
  #X=sm.add_constant(X)
  plt.plot(y,ols.predict(X),'go')
  plt.plot(y,y,'r--')
  plt.xlabel('E_B3LYP-E_0 (eV)')
  plt.ylabel('E_PRED-E_0 (eV)')
  plt.show()

  params=ols.params
  errs=  ols.bse
  data_row=np.concatenate((params.values,errs.values))
  data_row=np.concatenate((np.array([detgen[i],N[i],Ndet[i],c[i]]),data_row))
  if(data is None): data=data_row
  else: 
    if(len(data.shape)<2): data=data[:,np.newaxis]
    data=np.concatenate((data,data_row[:,np.newaxis]),axis=1)
exit(0)

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
      #x=j+np.random.normal(size=len(y),scale=0.1)
      x=j+np.linspace(-0.25,0.25,num=len(y))
      if(j==0):
        plt.errorbar(x,y,yerr=yerr,fmt=c[i]+marker[i],label='detgen='+detgen)
      else:
        plt.errorbar(x,y,yerr=yerr,fmt=c[i]+marker[i])
      j+=1

#ex_analysis1.pdf
plt.ylabel("Parameter value (eV)")
plt.xlabel("Parameter")
plt.xticks(np.arange(j),list(sub_df)[:10])
plt.axhline(0.0,color='k',ls='--')
plt.legend(loc='best')
plt.title("1-body fit, varying sampling schemes")
plt.ylim((-4,8))
plt.show()
#plt.savefig('ex_analysisF.pdf')
