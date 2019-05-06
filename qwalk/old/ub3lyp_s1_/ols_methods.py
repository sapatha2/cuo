import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.model_selection import KFold
from scipy import stats
from ed import ED
from roks_model import ED_roks
from uks_model import ED_uks
import itertools
from expectile import expectile_fit
from log import log_fit,log_fit_bootstrap
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce

######################################################################################
#OLS METHODS
#OLS ANALYSIS
'''
  ncv=5
  X=df[['mo_n_4s','mo_n_2ppi','mo_n_2pz','Jsd','Us']]
  hopping=df[['mo_t_pi','mo_t_dz','mo_t_ds','mo_t_sz']]
  y=df['energy']
  model_list=[]
  for n in range(1,hopping.shape[1]+1):
    s=set(list(hopping))
    models=list(map(set,itertools.combinations(s,n)))
    model_list+=[list(X)+list(m) for m in models]
  oneparm_valid(df,ncv,model_list,save=save)

  #ED_DMC + Beta
  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','Jsd','Us']
  dmc_eigenvalues = ed_dmc_beta(df,model,save=save) 
  dmc_eigenvalues['calc']='dmc'

  #ED_ROKS
  roks_eigenvalues = ED_roks(save=save)
  roks_eigenvalues['calc']='roks'

  #ED UKS
  uks_eigenvalues = ED_uks(save=save)
  uks_eigenvalues['calc']='uks'
'''
#Single parameter goodness of fit validation
def oneparm_valid(df,ncv,model_list,save=False,weights=None):
  zz=0
  kf=KFold(n_splits=ncv,shuffle=True)
  if(weights is None): weights=np.ones(df.shape[0])
  df['weights']=weights
  for model in model_list:
    print(model)
    y=df['energy']
    X=df[model]
    X=sm.add_constant(X)
   
    #10 samples of ncv
    r2_train=[]
    r2_test=[]
    r2=[]
    evals=[]
    for train_index,test_index in kf.split(df):
      X_train,X_test=X.iloc[train_index],X.iloc[test_index]
      y_train,y_test=y.iloc[train_index],y.iloc[test_index]
      w_train,w_test=df['weights'].iloc[train_index],df['weights'].iloc[test_index]
      ols=linear_model.LinearRegression().fit(X_train,y_train,w_train)
      r2_train.append(r2_score(y_train,ols.predict(X_train),w_train))
      r2_test.append(r2_score(y_test,ols.predict(X_test),w_test))
      r2.append(r2_score(y,ols.predict(X),weights))
    
    if(zz==0):
      plt.plot(np.ones(ncv*(1))*zz,r2_test,'gs-',label='r2 test')
      plt.plot(np.ones(ncv*(1))*zz+0.10,r2_train,'bo-',label='r2 train')
      plt.plot(np.ones(ncv*(1))*zz+0.20,r2,'r*-',label='full r2')
    else:
      plt.plot(np.ones(ncv*(1))*zz,r2_test,'gs-')
      plt.plot(np.ones(ncv*(1))*zz+0.10,r2_train,'bo-')
      plt.plot(np.ones(ncv*(1))*zz+0.20,r2,'r*-')
    zz+=1
  plt.legend(loc='best')
  plt.title('R2 CV DMC')
  plt.xlabel('Model')

  if(save):
    plt.savefig('analysis/cv_valid.pdf',bbox_inches='tight')
  else:
    plt.show()
  plt.close()
  return 1

#Regression plots
def regr_plot(df,model,weights=None,show=False):
  y=df['energy']
  Z=df[model]
  Z=sm.add_constant(Z)
  if(weights is None): ols=sm.OLS(y,Z).fit()
  else: ols=sm.WLS(y,Z,weights).fit() 
 
  #USE 500x Bootstrap to get CIs
  yhat=[]
  coef=[]
  df['weights']=weights
  for i in range(500):
    dfi=df.sample(n=df.shape[0],replace=True)
    #res_expi, __ = log_fit(dfi)
    #yhati = pred(res_expi,df.drop(columns=['energy','weights']))
    Zi=df[model]
    Zi=sm.add_constant(Zi)
    yhati = sm.OLS(dfi['energy'],Zi,weights=dfi['weights']).fit().predict()

    yhat.append(yhati)
  yhat=np.array(yhat)
  
  #Confidence intervals
  u_ols = np.percentile(yhat,97.5,axis=0)
  l_ols = np.percentile(yhat,2.5,axis=0)

  #USE analytic method
  #__,l_ols,u_ols=wls_prediction_std(ols,alpha=0.05) #Confidence level for two-sided hypothesis, 95 right now
  
  if(show):
    with open('analysis/fit.csv', 'w') as fh:
      fh.write(ols.summary().as_csv())
  else:
    print(ols.summary())

  df['pred']=ols.predict(Z)
  df['resid']=df['energy']-df['pred']
  df['pred_err']=(u_ols-l_ols)/2
  
  g = sns.FacetGrid(df,hue='Sz')
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  plt.title('Regression, 95% CI')
  plt.xlabel('Predicted Energy, eV')
  plt.ylabel('DMC Energy, eV')
  if(show):
    plt.show()
  plt.close()

  df=df[df['basestate']==-1]
  g = sns.FacetGrid(df,hue='Sz')
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  plt.title('Regression base states only, 95% CI')
  plt.xlabel('Predicted Energy, eV')
  plt.ylabel('DMC Energy, eV')
  if(show):
    plt.show()
  plt.close()

  return ols

#Exact diagonalization + beta fit 
def ed_dmc_beta(df,model,betas=np.arange(0,3.75,0.25),save=False):
  full_df=None
  for beta in betas:
    print("beta =============================================== "+str(beta))
    weights=np.exp(-beta*(df['energy']-min(df['energy'])))
    fit=regr_plot(df,model,weights,show=False)

    if(beta==2): 
      plt.plot(df['mo_n'],weights,'o')
      plt.title('Beta = 2')
      plt.xlabel('MO trace')
      plt.ylabel('Weight')
      if(save): plt.savefig('analysis/w_vs_t.pdf'); plt.close()
      else: plt.show(); plt.close()
      fit=regr_plot(df,model,weights,show=True)
    
    params=list(fit.params[1:5])+[0,0,0]+list(fit.params[5:])
    norb=9

    nelec=(8,7)
    nroots=14
    res1=ED(params,nroots,norb,nelec)

    nelec=(9,6)
    nroots=6
    res3=ED(params,nroots,norb,nelec)

    E = res1[0]
    n_occ = res1[2]+res1[3]
    Sz = np.ones(len(E))*0.5
    n_3d = n_occ[:,0] + n_occ[:,1] + n_occ[:,2] + n_occ[:,3] + n_occ[:,6]
    n_2ppi = n_occ[:,4] + n_occ[:,5]
    n_2pz = n_occ[:,7]
    n_4s = n_occ[:,8]
    d = pd.DataFrame({'E':E,'Sz':Sz,'n_3d':n_3d,'n_2pz':n_2pz,'n_2ppi':n_2ppi,'n_4s':n_4s})
    
    E = res3[0]
    n_occ = res3[2]+res3[3]
    Sz = np.ones(len(E))*1.5
    n_3d = n_occ[:,0] + n_occ[:,1] + n_occ[:,2] + n_occ[:,3] + n_occ[:,6]
    n_2ppi = n_occ[:,4] + n_occ[:,5]
    n_2pz = n_occ[:,7]
    n_4s = n_occ[:,8]
    d = pd.concat((d,pd.DataFrame({'E':E,'Sz':Sz,'n_3d':n_3d,'n_2pz':n_2pz,'n_2ppi':n_2ppi,'n_4s':n_4s})),axis=0)

    d['E']-=min(d['E'])
    d['eig']=np.arange(d.shape[0])
    d['beta']=beta
    if(full_df is None): full_df = d
    else: full_df = pd.concat((full_df,d),axis=0)

  sns.pairplot(full_df,vars=['E','n_2ppi','n_2pz','n_4s','n_3d'],hue='beta',markers=['s']+['o']*14)
  if(save): plt.savefig('analysis/beta_dmc_eigenvalues.pdf',bbox_inches='tight'); plt.close()
  else: plt.show(); plt.close()
  
  sns.pairplot(full_df,vars=['E','n_2ppi','n_2pz','n_4s','n_3d'],hue='Sz')
  if(save): plt.savefig('analysis/beta_dmc_eigenvalues_sz.pdf',bbox_inches='tight'); plt.close()
  else: plt.show(); plt.close()
  return full_df

