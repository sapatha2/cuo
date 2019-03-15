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
from ed import ED

######################################################################################
#FROZEN METHODS

#Collect df
def collect_df():
  df=None
  for basestate in range(11):
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
def oneparm_valid(df,ncv,model_list,save=False,weights=None):
  zz=0
  kf=KFold(n_splits=ncv,shuffle=True)
  if(weights is None): weights=np.ones(df.shape[0])
  df['weights']=weights
  for model in model_list:
    y=df['energy']
    X=df[model]
    X=sm.add_constant(X)
   
    #10 samples of ncv
    r2_train=[]
    r2_test=[]
    r2=[]
    evals=[]
    for i in range(10):
      for train_index,test_index in kf.split(df):
        X_train,X_test=X.iloc[train_index],X.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        w_train,w_test=df['weights'].iloc[train_index],df['weights'].iloc[test_index]
        ols=linear_model.LinearRegression().fit(X_train,y_train,w_train)
        r2_train.append(r2_score(y_train,ols.predict(X_train),w_train))
        r2_test.append(r2_score(y_test,ols.predict(X_test),w_test))
        r2.append(r2_score(y,ols.predict(X),weights))
    
    if(zz==0):
      plt.plot(np.ones(ncv*(i+1))*zz,r2_test,'gs-',label='r2 test')
      plt.plot(np.ones(ncv*(i+1))*zz+0.10,r2_train,'bo-',label='r2 train')
      plt.plot(np.ones(ncv*(i+1))*zz+0.20,r2,'r*-',label='full r2')
    else:
      plt.plot(np.ones(ncv*(i+1))*zz,r2_test,'gs-')
      plt.plot(np.ones(ncv*(i+1))*zz+0.10,r2_train,'bo-')
      plt.plot(np.ones(ncv*(i+1))*zz+0.20,r2,'r*-')
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
    
    density   = stats.kde.gaussian_kde(resid)
    density_1 = stats.kde.gaussian_kde(resid_1)
    density_2 = stats.kde.gaussian_kde(resid_2)
    x   = np.linspace(min(resid),max(resid),100)
    x_1 = np.linspace(min(resid_1),max(resid_1),100)
    x_2 = np.linspace(min(resid_2),max(resid_2),100)

    plt.subplot(211)
    plt.title('Histogram of residuals')
    plt.hist(resid_1,density=True,label='Sz=0.5') 
    plt.hist(resid_2,density=True,label='Sz=1.5')
    plt.hist(resid,density=True,label='combined') 
    
    plt.subplot(212)
    plt.title('Gaussian KDE of residuals')
    plt.plot(x_1,density_1(x_1),label='Sz=0.5') 
    plt.plot(x_2,density_2(x_2),label='Sz=1.5')
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
def regr_plot(df,model,weights=None,save=False):
  y=df['energy']
  Z=df[model]
  Z=sm.add_constant(Z)
  if(weights is None): ols=sm.OLS(y,Z).fit()
  else: ols=sm.WLS(y,Z,weights).fit() 
  
  __,l_ols,u_ols=wls_prediction_std(ols,alpha=0.05) #Confidence level for two-sided hypothesis, 95 right now
  if(save):
    with open('analysis/fit.csv', 'w') as fh:
      fh.write(ols.summary().as_csv())
  else:
    print(ols.summary())

  df['pred']=ols.predict(Z)
  df['resid']=df['energy']-df['pred']
  df['pred_err']=(u_ols-l_ols)/2
  '''
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
  '''
  return ols

#Weighted bootstrap sampling
def bootstrap(df,n,model,weights=None,save=False):
  if(weights is None): bootstrap=np.ones(df.shape[0])
  rdf = df.sample(n=n,weights=weights,replace=True) #Resampled df, using weights, with replacement
  regr_plot(rdf,model,save=save)
  
  return 1

######################################################################################
#Analysis pipeline, main thing to edit for runs
def analyze(df,save=False):
  ncv=5

  '''
  model_list=[
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz'],
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi'],
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_ds'],
   
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','Us'],
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','Us'],
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_ds','Us'],
 
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','Jsd'],
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','Jsd'],
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_ds','Jsd'],
  
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','Jsd','Us'],
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','Jsd','Us'],
    ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_ds','Jsd','Us'],
  ]
  oneparm_valid(df,ncv,model_list,save=save)
  '''
  
  '''
  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','Us']
  y=df['energy']
  X=df[model]
  X=sm.add_constant(X)
  df['resid']=df['energy']-sm.OLS(y,X).fit().predict()
  df['pred']=sm.OLS(y,X).fit().predict()
  sns.pairplot(df,vars=['energy','pred','resid','Jsd'],hue='Sz')
  plt.show()
  exit(0)
  '''

  #model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','Us']
  #regr_plot(df,model)
  #exit(0)

  '''
  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','Jcu','Us']
  #Logistical sigmoid with 1/2 cutoff specified
  def cut_sigmoid(y,cutoff,beta):
    return np.exp(-beta*(y-cutoff))/(np.exp(-beta*(y-cutoff))+1)

  zz=0
  ncv=10
  Nrun=1 #How many times to sample CV DISTRIBUTION 
  kf=KFold(n_splits=ncv,shuffle=True)
  for beta in np.arange(0,3.5,0.5):
    #weights=cut_sigmoid(-df['mo_n_3d']+max(df['mo_n_3d']),cut,beta)
    print(beta,"-----------------")
    y=df['energy']
    X=df[model]
    X=sm.add_constant(X)
    df['weights']=np.exp(-beta*(y-min(y)))
    weights=df['weights']
    
    r2_train=[]
    r2_test=[]
    r2=[]
    evals=[]
    for n in range(Nrun):
      for train_index,test_index in kf.split(df):
        X_train,X_test=X.iloc[train_index],X.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]
        train_weights=weights.iloc[train_index]
        test_weights= weights.iloc[test_index]

        ols=linear_model.LinearRegression().fit(X_train,y_train,train_weights)
        r2_train.append(r2_score(y_train,ols.predict(X_train),train_weights))
        r2_test.append(r2_score(y_test,ols.predict(X_test),test_weights))
        r2.append(r2_score(y,ols.predict(X),weights))
   
    plt.plot(np.ones(ncv*Nrun)*beta,    r2_train,'g.')
    plt.plot(np.ones(ncv*Nrun)*beta+0.1,r2_test,'b.')
    plt.plot(np.ones(ncv*Nrun)*beta+0.2,r2,'r*')
    zz+=1
  plt.title("cv, model="+' '.join(model))
  plt.show()
  '''

  full_df=None
  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','Jsd','Us']
  for beta in np.arange(0,3.75,0.25):
    weights=np.exp(-beta*(df['energy']-min(df['energy'])))
    fit=regr_plot(df,model,weights,save)
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
  sns.pairplot(full_df,vars=['E','n_2ppi','n_2pz','n_4s','n_3d'],hue='beta',markers=['o']+['.']*14)
  plt.show() 

if __name__=='__main__':
  df=collect_df()
  df=format_df(df)
  analyze(df,save=False)
