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
from scipy import stats

######################################################################################
#FROZEN METHODS

#Collect df
def collect_df():
  df=None
  for basestate in range(10):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)

      small_df['basestate']=basestate
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=0.5
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0,sort=True)

  for basestate in range(6):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='../ub3lyp_s3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=1.5
      df = pd.concat((df,small_df),axis=0,sort=True)
  
  for basestate in range(1,2):
    for gsw in [1.0]:
      f='../ub3lyp_do/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=-0.5
      df = pd.concat((df,small_df),axis=0,sort=True)
  
  return df

#Formatting
def format_df(df):
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
  df['mo_n']=df['mo_n_3d']+df['mo_n_2p']+df['mo_n_4s']

  df['Us']=df['u0']
  df['Ud']=df['u1']+df['u2']+df['u3']+df['u4']+df['u5']

  df['Jd']=np.zeros(df.shape[0])
  orb1=[1,1,1,1,2,2,2,3,3,4]
  orb2=[2,3,4,5,3,4,5,4,5,5]
  for i in range(len(orb1)):
    df['Jd']+=df['j_'+str(orb1[i])+'_'+str(orb2[i])]
  df['Jsd']=df['j_0_1']+df['j_0_2']+df['j_0_3']+df['j_0_4']+df['j_0_5']
  df['Jcu']=df['Jsd']+df['Jd']
  return df

#Single parameter goodness of fit validation
def oneparm_valid(df,ncv,model_list,save=False):
  zz=0
  kf=KFold(n_splits=ncv,shuffle=True)
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
    
    if(zz==0):
      plt.plot(np.ones(ncv)*zz,r2_test,'gs-',label='r2 test')
      plt.plot(np.ones(ncv)*zz+0.10,r2_train,'bo-',label='r2 train')
      plt.plot(np.ones(ncv)*zz+0.20,r2,'r*-',label='full r2')
    else:
      plt.plot(np.ones(ncv)*zz,r2_test,'gs-')
      plt.plot(np.ones(ncv)*zz+0.10,r2_train,'bo-')
      plt.plot(np.ones(ncv)*zz+0.20,r2,'r*-')
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

#Residual analysis, KDE and histogram
def resid_valid(df,model_list,save=False):
  zz=0
  for model in model_list:
    y=df['energy']
    X=df[model]
    X=sm.add_constant(X)
    ols=linear_model.LinearRegression().fit(X,y)

    resid = y - ols.predict(X)
    resid_1 = resid[df['Sz']==0.5]
    resid_2 = resid[df['Sz']==1.5]
    resid_3 = resid[df['Sz']==-0.5]

    density   = stats.kde.gaussian_kde(resid)
    density_1 = stats.kde.gaussian_kde(resid_1)
    density_2 = stats.kde.gaussian_kde(resid_2)
    density_3 = stats.kde.gaussian_kde(resid_3)
    x   = np.linspace(min(resid),max(resid),100)
    x_1 = np.linspace(min(resid_1),max(resid_1),100)
    x_2 = np.linspace(min(resid_2),max(resid_2),100)
    x_3 = np.linspace(min(resid_3),max(resid_3),100)

    plt.subplot(211)
    plt.title('Histogram of residuals')
    plt.hist(resid_1,density=True,label='Sz=0.5') 
    plt.hist(resid_2,density=True,label='Sz=1.5')
    plt.hist(resid_3,density=True,label='Sz=0.5, do')
    plt.hist(resid,density=True,label='combined') 
    
    plt.subplot(212)
    plt.title('Gaussian KDE of residuals')
    plt.plot(x_1,density_1(x_1),label='Sz=0.5') 
    plt.plot(x_2,density_2(x_2),label='Sz=1.5')
    plt.plot(x_3,density_2(x_3),label='Sz=0.5, do')
    plt.plot(x,density(x),label='combined') 
   
    plt.suptitle('Model ='+' '.join(model))
    plt.legend(loc='best')
     
    if(save):
      plt.savefig('analysis/resid_analysis_'+str(zz)+'.pdf',bbox_inches='tight')
    else:
      plt.show()
    plt.close()
    zz+=1
  return 1

#Regression plots
def regr_plot(df,model,save=False):
  y=df['energy']
  Z=df[model]
  Z=sm.add_constant(Z)
  ols=sm.OLS(y,Z).fit() 
  
  __,l_ols,u_ols=wls_prediction_std(ols,alpha=0.05) #Confidence level for two-sided hypothesis, 95 right now
  if(save):
    with open('analysis/fit.csv', 'w') as fh:
      fh.write(ols.summary().as_csv())
  else:
    print(ols.summary())

  df['pred']=ols.predict(Z)
  df['resid']=df['energy']-df['pred']
  df['pred_err']=(u_ols-l_ols)/2
  #sns.pairplot(df,vars=['energy','pred']+model,hue='Sz')#,hue='basestate',markers=['o']+['.']*6)
  if(save):
    plt.savefig('analysis/fit_pairplot.pdf',bbox_inches='tight')
  else:
    plt.show()
  plt.close()

  g = sns.FacetGrid(df,hue='Sz',hue_kws=dict(marker=['.']*3))#,hue='basestate',hue_kws=dict(marker=['o']+['.']*16))
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  if(save):
    plt.savefig('analysis/fit.pdf',bbox_inches='tight')
  else:
    plt.show()
  plt.close()

  df=df[df['basestate']==-1]
  g = sns.FacetGrid(df,hue='Sz',hue_kws=dict(marker=['.']*3))#,hue='basestate',hue_kws=dict(marker=['o']+['.']*16))
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  if(save):
    plt.savefig('analysis/fit_baseonly.pdf',bbox_inches='tight')
  else:
    plt.show()
  plt.close()
  return 1

######################################################################################
#Analysis pipeline, main thing to edit for runs
def analyze(df,save=False):
  '''
  ncv=10
  model_list=[
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_ds'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_ds'],
    
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','Jcu'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','Jsd'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_ds','Jsd'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_ds','Jsd'],
  
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','Jcu','Us'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','Jsd','Us'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_ds','Jsd','Us'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_ds','Jsd','Us'],
  
  ]
  oneparm_valid(df,ncv,model_list,save=save)
  
  model_list=[
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_ds'],
    
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','Jsd'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_ds','Jsd'],
  
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','Jsd','Us'],
    ['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_ds','Jsd','Us'],
  ]
  resid_valid(df,model_list,save=save)
  '''
  model=['mo_n_3d','mo_n_2ppi','mo_n_2pz','mo_t_pi','Jsd','Us']
  regr_plot(df,model,save=save)

if __name__=='__main__':
  df=collect_df()
  df=format_df(df)
  analyze(df,save=False)
