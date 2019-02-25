import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from diagonalize import diagonalize, new_gs
from statsmodels.sandbox.regression.predstd import wls_prediction_std

def collectdf():
  df=None
  for basestate in range(10):
    for gsw in np.arange(0.1,1.01,0.1):
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/gosling.pickle' 
      small_df=pd.read_pickle(f)
      
      small_df['basestate']=basestate
      if(np.around(gsw,2)==1.0): small_df['basestate']=-1
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0)
  return df

def analyze(df):
  #Formatting
  df['gsw']=np.round(df['gsw'],2)
  df['n_3dd']=df['t_8_8']+df['t_9_9']
  df['n_3dpi']=df['t_5_5']+df['t_6_6']
  df['n_3dz2']=df['t_7_7']
  df['n_3d']=df['n_3dd']+df['n_3dpi']+df['n_3dz2']
  df['n_2ppi']=df['t_10_10']+df['t_11_11']
  df['n_2pz']=df['t_12_12']
  df['n_2p']=df['n_2ppi']+df['n_2pz']
  df['n_4s']=df['t_13_13']
  df['t_pi']=2*(df['t_5_10']+df['t_6_11'])
  df['t_dz']=2*df['t_7_12']
  df['t_ds']=2*df['t_7_13']
  df['t_sz']=2*df['t_12_13']

  #PAIRPLOTS --------------------------------------------------------------------------
  #sns.pairplot(df,vars=['energy','n_3dd','n_3dpi','n_3dz2','n_3d'],hue='basestate',markers=['o']+['.']*10)
  #sns.pairplot(df,vars=['energy','n_2ppi','n_2pz','n_2p','n_4s'],hue='basestate',markers=['o']+['.']*10)
  #sns.pairplot(df,vars=['energy','t_pi','t_dz','t_ds','t_sz'],hue='basestate',markers=['o']+['.']*10)
  #plt.show()
  #exit(0)

  #R2, RMSE AND MODEL PLOTS ----------------------------------------------------------
  '''zz=0
  ncv=5
  model_list=[
    ['n_3d','n_2ppi','n_2pz'],
    ['n_3d','n_2ppi','n_2pz','t_pi'],
    ['n_3d','n_2ppi','n_2pz','t_ds'],
    ['n_3d','n_2ppi','n_2pz','t_pi','t_ds'],
    ['n_3d','n_2ppi','n_2pz','t_pi','t_ds','t_dz'],
    ['n_3d','n_2ppi','n_2pz','t_pi','t_ds','t_sz'],
    ['n_3d','n_2ppi','n_2pz','t_pi','t_ds','t_dz','t_sz']
  ]
  for model in model_list:
    y=df['energy']
    X=df[model]
    X=sm.add_constant(X)
    ols=linear_model.LinearRegression().fit(X,y)
    
    full_r2=r2_score(y,ols.predict(X))
    full_rmse=mean_squared_error(y,ols.predict(X))
    cv_r2=cross_val_score(ols,X,y,cv=ncv,scoring='r2')
    cv_rmse=cross_val_score(ols,X,y,cv=ncv,scoring='neg_mean_squared_error')
    
    coefs=ols.coef_[1:]
    parms=[0]
    zzz=0
    for i in range(len(model_list[-1])):
      if(model_list[-1][i] in model):
        parms.append(coefs[zzz])
        zzz+=1
      else: parms.append(0)
    
    evals=diagonalize(parms)
    #if(zz==3): new_gs(parms) 
    
    plt.plot(np.ones(ncv)*zz,cv_r2,'go-')
    plt.plot(np.ones(ncv)*zz+0.15,cv_rmse,'bo-')
    
    if(zz==0):
      plt.plot([zz],full_r2,'gs',label='R2')
      plt.plot([zz+0.15],full_rmse,'bs',label='RMSE')
      plt.plot(np.ones(len(evals))*zz+0.30,evals/10,'ro-',label='Eval (eV)/10')
    else:
      plt.plot([zz],full_r2,'gs')
      plt.plot([zz+0.15],full_rmse,'bs')
      plt.plot(np.ones(len(evals))*zz+0.30,evals/10,'ro-')

    zz+=1
  #plt.legend(loc='best')
  #plt.xlabel('Model')
  #plt.savefig('model_valid.pdf',bbox_inches='tight')
  #exit(0)
  '''

  #PREDICTION PLOTS ---------------------------------------------------------------
  X=df[['n_3d','n_2ppi','n_2pz','t_pi','t_ds']]
  X=sm.add_constant(X)
  y=df['energy']
  ols=sm.OLS(y,X).fit() 
  __,l_ols,u_ols=wls_prediction_std(ols,alpha=0.05) #Confidence level for two-sided hypothesis, 95 right now

  df['pred_err']=(u_ols-l_ols)/2
  df['pred']=ols.predict(X)

  df=df[df['basestate']==-1]
  g = sns.FacetGrid(df,hue='basestate',hue_kws=dict(marker=['o']+['.']*10))
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  plt.show()
  exit(0)
if __name__=='__main__':
  df=collectdf()
  analyze(df)
