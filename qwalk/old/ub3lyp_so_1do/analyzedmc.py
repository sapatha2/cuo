import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
#from diagonalize import diagonalize, new_gs
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.model_selection import KFold

def collectdf():
  df=None
  for basestate in range(10):
    for gsw in np.arange(0.1,1.01,0.1):
      f='../ub3lyp_mo3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
     
      small_df['Sz']=0.5
      small_df['basestate']=basestate
      if(np.around(gsw,2)==1.0): small_df['basestate']=-1
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0)
  
  for basestate in range(6):
    for gsw in np.arange(0.1,1.01,0.1):
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
     
      small_df['Sz']=1.5
      small_df['basestate']=basestate+10
      if(np.around(gsw,2)==1.0): small_df['basestate']=-1
      df = pd.concat((df,small_df),axis=0)
  return df

def analyze(df):
  #Formatting
  df['gsw']=np.round(df['gsw'],2)
  df['n_3dd']=df['t_4_4']+df['t_5_5']
  df['n_3dpi']=df['t_1_1']+df['t_2_2']
  df['n_3dz2']=df['t_3_3']
  df['n_3d']=df['n_3dd']+df['n_3dpi']+df['n_3dz2']
  df['n_2ppi']=df['t_6_6']+df['t_7_7']
  df['n_2pz']=df['t_8_8']
  df['n_2p']=df['n_2ppi']+df['n_2pz']
  df['n_4s']=df['t_9_9']
  df['t_pi']=2*(df['t_1_6']+df['t_2_7'])
  df['t_dz']=2*df['t_3_8']
  df['t_sz']=2*df['t_8_9']
  df['t_ds']=2*df['t_3_9']
  df['n']=df['n_3d']+df['n_4s']+df['n_2p']

  df=df[df['basestate']==-1]
  #print(df.iloc[[2,10,4,11]][['energy','n_2pz','n_2ppi','n_4s','t_pi','t_ds','t_dz','t_sz']])
  #print(df.var())
  #exit(0)

  #PAIRPLOTS --------------------------------------------------------------------------
  #sns.pairplot(df,vars=['energy','n_3dd','n_3dpi','n_3dz2','n_3d'],hue='basestate',markers=['o']+['.']*10)
  sns.pairplot(df,vars=['energy','n_2ppi','n_2pz','n_2p','n_4s'],hue='basestate',markers=['o']+['.']*10)
  #sns.pairplot(df,vars=['energy','t_pi','t_dz','t_ds','t_sz'],hue='basestate',markers=['o']+['.']*10)
  #sns.pairplot(df,vars=['energy','n'],hue='Sz',markers=['o','o']) 
  plt.show()
  exit(0)

  #R2, RMSE AND MODEL PLOTS ----------------------------------------------------------
  zz=0
  ncv=5
  kf=KFold(n_splits=ncv,shuffle=True)
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

    r2_train=[]
    r2_test=[]
    r2=[]
    evals=[]
    for train_index,test_index in kf.split(df):
      X_train,X_test=X.iloc[train_index],X.iloc[test_index]
      y_train,y_test=y.iloc[train_index],y.iloc[test_index]

      ols=linear_model.LinearRegression().fit(X_train,y_train)
      r2_train.append(r2_score(y_train,ols.predict(X_train)))
      r2_test.append(r2_score(y_test,ols.predict(X_test)))
      r2.append(r2_score(y,ols.predict(X)))

      coefs=ols.coef_[1:]
      parms=[0]
      zzz=0
      for i in range(len(model_list[-1])):
        if(model_list[-1][i] in model):
          parms.append(coefs[zzz])
          zzz+=1
        else: parms.append(0)
      evals.append(diagonalize(parms))
    
    print(coefs)    
    plt.plot(np.ones(ncv)*zz,r2_test,'gs-')
    plt.plot(np.ones(ncv)*zz+0.10,r2_train,'bo-')
    plt.plot(np.ones(ncv)*zz+0.20,r2,'r*-')
    for pp in range(len(evals)):
      plt.plot(np.ones(len(evals[pp]))*zz+0.30+pp*0.10,evals[pp]/100+0.94,'ko-')
    zz+=1
  plt.legend(loc='best')
  plt.xlabel('Model')
  plt.savefig('dmc_model_valid.pdf',bbox_inches='tight')
  #plt.close()
  #plt.show()
  exit(0)

  #PREDICTION PLOTS ---------------------------------------------------------------
  X=df[['n_3d','n_2ppi','n_2pz','t_pi','t_dz']]
  #X=df[['n_3dz2','n_3dpi','n_3dd','n_2ppi','n_2pz','t_pi','t_ds']] 
  #X=df[['n_3d','n_2ppi','n_2pz','t_pi','t_ds']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_pi']]
  X=sm.add_constant(X)
  y=df['energy']
  ols=sm.OLS(y,X).fit() 
  __,l_ols,u_ols=wls_prediction_std(ols,alpha=0.05) #Confidence level for two-sided hypothesis, 95 right now
  print(ols.summary())

  df['pred_err']=(u_ols-l_ols)/2
  df['pred']=ols.predict(X)

  g = sns.FacetGrid(df,hue='Sz',hue_kws=dict(marker=['.']*2))#,hue='basestate',hue_kws=dict(marker=['o']+['.']*16))
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  plt.savefig('dmc_fit.pdf')
  plt.close()

  df=df[df['basestate']==-1]
  g = sns.FacetGrid(df,hue='Sz',hue_kws=dict(marker=['.']*2))#,hue='basestate',hue_kws=dict(marker=['o']+['.']*16))
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  plt.savefig('dmc_fit_baseonly.pdf')

if __name__=='__main__':
  df=collectdf()
  analyze(df)
