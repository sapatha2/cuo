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
  return format_df_iao(df)

#Get IAO psums
def format_df_iao(df):
  df['beta']=-1000
  #LOAD IN IAOS
  act_iao=[5,9,6,8,11,12,7,13,1]
  iao=np.load('../../pyscf/ub3lyp_full/b3lyp_iao_b.pickle')
  iao=iao[:,act_iao]
  
  #LOAD IN MOS
  act_mo=[5,6,7,8,9,10,11,12,13]
  chkfile='../../pyscf/chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  mo=m.mo_coeff[:,act_mo]
  s=m.get_ovlp()

  #IAO ordering: ['del','del','yz','xz','x','y','z2','z','s']
  #MO ordering:  dxz, dyz, dz2, delta, delta, px, py, pz, 4s
  
  df['iao_n_3d']=0
  df['iao_n_2pz']=0
  df['iao_n_2ppi']=0
  df['iao_n_4s']=0
  df['iao_t_pi']=0
  df['iao_t_dz']=0
  df['iao_t_ds']=0
  df['iao_t_sz']=0

  for z in range(df.shape[0]):
    print(z)
    e=np.zeros((9,9))
    orb1=[1,2,3,4,5,6,7,8,9,1,2,3,8,3]
    orb2=[1,2,3,4,5,6,7,8,9,6,7,8,9,9]
    for i in range(len(orb1)):
      e[orb1[i]-1,orb2[i]-1]=df['mo_'+str(orb1[i])+'_'+str(orb2[i])].values[z]
    
    mo_to_iao = reduce(np.dot,(mo.T,s,iao))
    e = reduce(np.dot,(mo_to_iao.T,e,mo_to_iao))
    e[np.abs(e)<1e-10]=0
    e=(e+e.T)/2

    df['iao_n_3d'].iloc[z]=np.sum(np.diag(e)[[0,1,2,3,6]])
    df['iao_n_2pz'].iloc[z]=np.sum(np.diag(e)[7])
    df['iao_n_2ppi'].iloc[z]=np.sum(np.diag(e)[[4,5]])
    df['iao_n_4s'].iloc[z]=np.sum(np.diag(e)[8])
    df['iao_t_pi'].iloc[z]=2*(e[3,4]+e[2,5])
    df['iao_t_ds'].iloc[z]=2*e[6,8]
    df['iao_t_dz'].iloc[z]=2*e[6,7]
    df['iao_t_sz'].iloc[z]=2*e[7,8]

  return df

######################################################################################
#OLS METHODS

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
  
  __,l_ols,u_ols=wls_prediction_std(ols,alpha=0.05) #Confidence level for two-sided hypothesis, 95 right now
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

######################################################################################
#LOG METHODS

#Single parameter goodness of fit validation
def oneparm_valid_log(df,ncv,model,weights=None):
  X=df[model+['energy']]
  X=sm.add_constant(X)
  if(weights is None): weights=np.ones(df.shape[0])
  X['weights']=weights

  exp_parms, yhat, yerr_u, yerr_l = log_fit_bootstrap(X)
  df['pred']=yhat
  df['pred_err']=(yerr_u - yerr_l)/2
  
  #10 samples of ncv
  kf=KFold(n_splits=ncv,shuffle=True)
  r2_train=[]
  r2_test=[]
  r2=[]
  evals=[]
  for train_index,test_index in kf.split(X):
    X_train,X_test=X.iloc[train_index],X.iloc[test_index]
    exp_parms, yhat, yerr_u, yerr_l = log_fit_bootstrap(X_train)
    exp_parms = np.mean(exp_parms,axis=0)

    y_train = X_train['energy']
    y_test = X_test['energy']
    w_train = X_train['weights']
    w_test = X_test['weights']
    yhat_train = np.dot(exp_parms,X_train.drop(columns=['energy','weights']).values.T)
    yhat_test = np.dot(exp_parms,X_test.drop(columns=['energy','weights']).values.T)

    r2_train.append(r2_score(yhat_train,y_train,w_train))
    r2_test.append(r2_score(yhat_test,y_test,w_test))
  r2_train=np.array(r2_train)
  r2_test=np.array(r2_test)
  return [np.mean(r2_train),np.std(r2_train),np.mean(r2_test),np.std(r2_test)]

#Log regression + plots
def regr_log_plot(df,model,weights=None,n=500,show=False,fname=None):
  X=df[model+['energy']]
  X=sm.add_constant(X)
  if(weights is None): weights=np.ones(df.shape[0])
  X['weights']=weights
  
  exp_parms, yhat, yerr_u, yerr_l = log_fit_bootstrap(X,n)
  df['pred']=yhat
  df['pred_err']=(yerr_u - yerr_l)/2

  g = sns.FacetGrid(df,hue='Sz')
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='.').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  plt.title('Regression')
  plt.xlabel('Predicted Energy, eV')
  plt.ylabel('DMC Energy, eV')
  if(show): pass
  else: plt.savefig(fname+'.pdf',bbox_inches='tight')
  plt.close()

  df=df[df['basestate']==-1]
  g = sns.FacetGrid(df,hue='Sz')
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='.').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  plt.title('Regression base states only')
  plt.xlabel('Predicted Energy, eV')
  plt.ylabel('DMC Energy, eV')
  if(show): pass 
  else: plt.savefig(fname+'_baseonly.pdf',bbox_inches='tight')
  plt.close()

  return exp_parms, yhat, (yerr_u - yerr_l)/2

#Beta regression
def regr_beta_log(df,model_list,betas=np.arange(0,3.75,0.25),save=False):
  full_df=None
  ncv=5

  zz=0
  for model in model_list:
    zz+=1
    for beta in betas:
      print("beta =============================================== "+str(beta))
      weights=np.exp(-beta*(df['energy']-min(df['energy'])))
      exp_parms, yhat, yerr=regr_log_plot(df,model,weights,show=(not save),fname='analysis/dmc_fit_log_beta'+str(beta)+'_'+str(zz))
      exp_parms = np.mean(exp_parms,axis=0)

      #Figure out which parameters are in my list
      param_names=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz',
      'mo_t_ds','mo_t_sz','Jsd','Us']
      params=[]
      for parm in param_names:
        if(parm in model): params.append(exp_parms[model.index(parm)+1])
        else: params.append(0)

      param_names=['beta']+param_names+['R2cv_mu_train','R2cv_std_train','R2cv_mu_test','R2cv_std_test']
      params=[beta]+params+oneparm_valid_log(df,ncv,model,weights)
      d=pd.DataFrame(data=np.array(params)[np.newaxis,:],columns=param_names,index=[0])
      if(full_df is None): full_df=d
      else: full_df = pd.concat((full_df,d),axis=0)
      print(full_df)
  return full_df

#Plot single parameter goodness of fit 
def plot_valid_log(save=False):
  full_df=pd.read_pickle('analysis/regr_beta_log.pickle')
  model=[]
  for i in range(15):
    model+=list(np.linspace(i,i+0.75,15))#[i]*15
  full_df['model']=model

  '''
  print(full_df[full_df['model']==2].iloc[0])
  print(full_df[full_df['model']==5].iloc[0])
  print(full_df[full_df['model']==7].iloc[0])
  print(full_df[full_df['model']==9].iloc[0])
  print(full_df[full_df['model']==10].iloc[0])
  print(full_df[full_df['model']==12].iloc[0])
  print(full_df[full_df['model']==13].iloc[0])
  print(full_df[full_df['model']==14].iloc[0])
  exit(0)
  '''
  
  g = sns.FacetGrid(full_df,hue='beta')
  g.map(plt.errorbar, "model", "R2cv_mu_train", "R2cv_std_train",fmt='.').add_legend()
  plt.xlabel('Model, eV')
  plt.ylabel('R2cv_train, eV')
  if(save): plt.savefig('analysis/oneparm_valid_log_train.pdf',bbox_inches='tight')
  else: plt.show()
  plt.close()

  g = sns.FacetGrid(full_df,hue='beta')
  g.map(plt.errorbar, "model", "R2cv_mu_test", "R2cv_std_test",fmt='.').add_legend()
  plt.xlabel('Model, eV')
  plt.ylabel('R2cv_test, eV')
  if(save): plt.savefig('analysis/oneparm_valid_log_test.pdf',bbox_inches='tight')
  else: plt.show()
  plt.close()

#Exact diagonalization using log regression
def ed_dmc_beta_log(df,model,betas=np.arange(0,3.75,0.25),save=False,fname=None):
  full_df=None
  for beta in betas:
    print("beta =============================================== "+str(beta))
    weights=np.exp(-beta*(df['energy']-min(df['energy'])))
    exp_parms, yhat, yerr=regr_log_plot(df,model,weights,show=save)
    exp_parms = np.mean(exp_parms,axis=0)

    #Figure out which parameters are in my list
    param_names=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz',
    'mo_t_ds','mo_t_sz','Jsd','Us']
    params=[]
    for parm in param_names:
      if(parm in model): params.append(exp_parms[model.index(parm)+1])
      else: params.append(0)
  
    print(param_names)
    print(params)
    
    norb=9
    nelec=(8,7)
    nroots=14
    res1=ED(params,nroots,norb,nelec)

    nelec=(9,6)
    nroots=6
    res3=ED(params,nroots,norb,nelec)

    E = res1[0]
    Sz = np.ones(len(E))*0.5
    dm = res1[2] + res1[3]
    n_3d = dm[:,0,0]+dm[:,1,1]+dm[:,2,2]+dm[:,3,3]+dm[:,6,6]
    n_2ppi = dm[:,4,4]+dm[:,5,5]
    n_2pz = dm[:,7,7]
    n_4s = dm[:,8,8]
    t_pi = 2*(dm[:,3,4]+dm[:,2,5])
    t_ds = 2*dm[:,6,8]
    t_dz = 2*dm[:,6,7]
    t_sz = 2*dm[:,7,8]
    d = pd.DataFrame({'energy':E,'Sz':Sz,'iao_n_3d':n_3d,'iao_n_2pz':n_2pz,'iao_n_2ppi':n_2ppi,'iao_n_4s':n_4s,
    'iao_t_pi':t_pi,'iao_t_ds':t_ds,'iao_t_dz':t_dz,'iao_t_sz':t_sz})

    E = res3[0]
    Sz = np.ones(len(E))*1.5
    dm = res3[2] + res3[3]
    n_3d = dm[:,0,0]+dm[:,1,1]+dm[:,2,2]+dm[:,3,3]+dm[:,6,6]
    n_2ppi = dm[:,4,4]+dm[:,5,5]
    n_2pz = dm[:,7,7]
    n_4s = dm[:,8,8]
    t_pi = 2*(dm[:,3,4]+dm[:,2,5])
    t_ds = 2*dm[:,6,8]
    t_dz = 2*dm[:,6,7]
    t_sz = 2*dm[:,7,8]
    d = pd.concat((d,pd.DataFrame({'energy':E,'Sz':Sz,'iao_n_3d':n_3d,'iao_n_2pz':n_2pz,'iao_n_2ppi':n_2ppi,'iao_n_4s':n_4s,
    'iao_t_pi':t_pi,'iao_t_ds':t_ds,'iao_t_dz':t_dz,'iao_t_sz':t_sz})),axis=0)

    d['energy']-=min(d['energy'])
    d['eig']=np.arange(d.shape[0])
    d['beta']=beta

    if(full_df is None): full_df = d
    else: full_df = pd.concat((full_df,d),axis=0)

  #Spin only and number occupations
  sns.pairplot(full_df,vars=['energy','iao_n_2ppi','iao_n_2pz','iao_n_4s','iao_n_3d'],hue='Sz')
  if(save): plt.savefig(fname+'_sz.pdf',bbox_inches='tight'); plt.close()
  else: plt.show(); plt.close()

  #All parameters AND sampled states
  df['energy']-=min(df['energy'])
  z_df = pd.concat((full_df,df[['energy','iao_n_2ppi','iao_n_2pz','iao_n_4s','iao_n_3d','iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz','beta','Sz']]),axis=0)
  #sns.pairplot(z_df,vars=['energy','iao_n_2ppi','iao_n_2pz','iao_n_4s','iao_n_3d','iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz'],hue='beta',markers=['.']+['s']+['o']*14)
  sns.pairplot(z_df,vars=['energy','iao_n_2ppi','iao_n_2pz','iao_n_4s','iao_n_3d','iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz'],hue='beta',markers=['.']+['o']) #Beta=2
  if(save): plt.savefig(fname+'.pdf',bbox_inches='tight'); plt.close()
  else: plt.show(); plt.close()
  
  return full_df

######################################################################################
#Analysis pipeline, main thing to edit for runs
def analyze(df,save=False):
  #One paramter validation for different included hoppings, OLS
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

  #Generate all possible models
  X=df[['mo_n_4s','mo_n_2ppi','mo_n_2pz','Jsd','Us']]
  hopping=df[['mo_t_pi','mo_t_dz','mo_t_ds','mo_t_sz']]
  y=df['energy']
  model_list=[]
  for n in range(1,hopping.shape[1]+1):
    s=set(list(hopping))
    models=list(map(set,itertools.combinations(s,n)))
    model_list+=[list(X)+list(m) for m in models]

  #Plotting regr + creating 1parm valid database log
  #full_df = regr_beta_log(df,model_list,save=save) 
  #full_df.to_pickle('analysis/regr_beta_log.pickle')
  
  #Plotting 1 parm valid database log
  #plot_valid_log(save)

  #ED + Plots of eigenvalues/eigenvectors
  eig_df=None
  for i in np.arange(15):
    model=model_list[i]
    d=ed_dmc_beta_log(df,model,betas=[2.0],save=save,fname='analysis/ed_dmc_beta_log_'+str(i))
    d['model']=i
    if(eig_df is None): eig_df = d
    else: eig_df = pd.concat((eig_df, d),axis=0)
  eig_df.to_pickle('analysis/ed_gosling.pickle')

if __name__=='__main__':
  #df=collect_df()
  #df=format_df(df)
  #df.to_pickle('formatted_gosling.pickle')
  #exit(0)

  #df=pd.read_pickle('formatted_gosling.pickle')
  #analyze(df,save=True)

  #df=pd.read_pickle('analysis/ed_gosling.pickle')
  #print(df.shape[0])
