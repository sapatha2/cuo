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
  for basestate in range(3):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)

      small_df['basestate']=basestate
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=0.5
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0,sort=True)

  for basestate in range(3):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='../ub3lyp_min_s3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=1.5
      df = pd.concat((df,small_df),axis=0,sort=True)
  return df

def analyze(df):
  #Formatting
  df['mo_n_3dd']=df['mo_4_4']+df['mo_5_5']
  df['mo_n_3dpi']=df['mo_1_1']+df['mo_2_2']
  df['mo_n_3dz2']=df['mo_3_3']
  df['mo_n_3d']=df['mo_n_3dd']+df['mo_n_3dpi']+df['mo_n_3dz2']
  df['mo_n_2ppi']=df['mo_6_6']+df['mo_7_7']
  df['mo_n_2pz']=df['mo_8_8']
  df['mo_n_2p']=df['mo_n_2ppi']+df['mo_n_2pz']
  df['mo_n_4s']=df['mo_9_9']
  df['mo_t_pi']=2*(df['mo_1_6']+df['mo_2_7'])
  df['mo_t_dz']=2*df['mo_3_8']
  df['mo_t_sz']=2*df['mo_8_9']
  df['mo_t_ds']=2*df['mo_3_9']

  df['Us']=df['u0']
  df['Ud']=df['u1']+df['u2']+df['u3']+df['u4']+df['u5']

  df['Jd']=np.zeros(df.shape[0])
  orb1=[1,1,1,1,2,2,2,3,3,4]
  orb2=[2,3,4,5,3,4,5,4,5,5]
  for i in range(len(orb1)):
    df['Jd']+=df['j_'+str(orb1[i])+'_'+str(orb2[i])]
  df['Jsd']=df['j_0_1']+df['j_0_2']+df['j_0_3']+df['j_0_4']+df['j_0_5']
  df['Jcu']=df['Jsd']+df['Jd']

  #plt.errorbar(np.ones(df.shape[0]),df['energy'],yerr=df['energy_err'],fmt='o')
  #plt.show()
  #exit(0)

  #sns.pairplot(df,vars=['energy','mo_n_2ppi','mo_n_2pz','mo_n_3d','Jsd'],hue='basestate',markers=['o']+['.']*3)
  #plt.show()
  #exit(0)
  #print(df[['energy','Jsd','Sz']])
  #exit(0)

  #VMC ordering of base states
  '''
  df['energy']-=min(df['energy'])
  df=df.sort_values(by=['energy'])
  df['order']=np.arange(df.shape[0])
  #df=df.iloc[6:]
  sns.pairplot(df,vars=['energy','order','n_3dz2','n_3dpi','n_3dd','n_2ppi','n_2pz','mo_pi','mo_dz','mo_sz','mo_ds'],hue='Sz')
  plt.savefig('VMC_order.pdf')
  exit(0)
  '''

  #PAIRPLOTS --------------------------------------------------------------------------
  '''
  ind=np.argsort(df['energy'])
  df=df.iloc[ind[:6]]
  #sns.pairplot(df,vars=['energy','mo_n_3d','mo_n_2ppi','mo_n_2pz','Jsd','Jsp'],hue='Sz')
  #plt.savefig('DMC.pdf',bbox_inches='tight')
  #sns.pairplot(df,vars=['energy','iao_n_3d','iao_n_2ppi','iao_n_2pz','Jsd','Jsp'],hue='Sz')
  #plt.show()
  #plt.close()
  #exit(0)
  '''

  '''
  y=df['energy']
  X=df['mo_n_3d']
  X=sm.add_constant(X)
  ols=sm.OLS(y,X).fit()
  print(ols.summary())
  df['mo_resid']=df['energy']-ols.predict(X)
  sns.pairplot(df,vars=['energy','mo_resid','Jsd','Jsp'],hue='Sz')
  plt.show()
  plt.close()

  y=df['energy']
  X=df['iao_n_3d']
  X=sm.add_constant(X)
  ols=sm.OLS(y,X).fit()
  print(ols.summary())
  df['iao_resid']=df['energy']-ols.predict(X)
  sns.pairplot(df,vars=['energy','iao_resid','Jsd','Jsp'],hue='Sz')
  plt.show()
  exit(0)
  '''

  #R2, RMSE AND MODEL PLOTS ----------------------------------------------------------
  '''
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
  plt.savefig('model_valid.pdf',bbox_inches='tight')
  #plt.close()
  #plt.show()
  exit(0)
  '''

  #PREDICTION PLOTS ---------------------------------------------------------------
  model=['mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_n_3d','Jsd']

  y=df['energy']
  Z=df[model]
  Z=sm.add_constant(Z)
  ols=sm.OLS(y,Z).fit() 
  
  #__,l_ols,u_ols=wls_prediction_std(ols,alpha=1.0) #Confidence level for two-sided hypothesis, 95 right now
  print(ols.summary())

  df['pred']=ols.predict(Z)
  df['resid']=df['energy']-df['pred']

  g = sns.FacetGrid(df,hue='Sz',hue_kws=dict(marker=['.']*3))#,hue='basestate',hue_kws=dict(marker=['o']+['.']*16))
  g.map(plt.errorbar, "pred", "energy", "energy_err","energy_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  #plt.show()
  plt.savefig('fit_Jsd.pdf')
  exit(0)

  df=df[df['basestate']==-1]
  g = sns.FacetGrid(df,hue='Sz',hue_kws=dict(marker=['.']*3))#,hue='basestate',hue_kws=dict(marker=['o']+['.']*16))
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  #plt.savefig('fit_baseonly.pdf')
  plt.show()

if __name__=='__main__':
  df=collectdf()
  analyze(df)
