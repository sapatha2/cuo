import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model, preprocessing
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
from log import log_fit,log_fit_bootstrap, rmse_bar
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import linear_sum_assignment
import scipy 
from find_connect import  *
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.odr import *
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
    
      small_df['basestate']=basestate+11
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=1.5
      df = pd.concat((df,small_df),axis=0,sort=True)

  '''
  for basestate in range(4):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='../../ub3lyp_extra_1/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+17
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=0.5
      df = pd.concat((df,small_df),axis=0,sort=True)
 
  for basestate in range(2):
    for gsw in np.arange(-1.0,1.2,0.2):
      f='../../ub3lyp_extra_3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+21
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=1.5
      df = pd.concat((df,small_df),axis=0,sort=True)
  '''

  '''
  for basestate in range(4):
    for gsw in np.arange(0.2,1.2,0.2):
      f='../../ub3lyp_extra_extra/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+23
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=0.5
      df = pd.concat((df,small_df),axis=0,sort=True)

  for basestate in np.arange(3):
    for gsw in np.arange(0.2,1.2,0.2):
      f='../../ub3lyp_extra_extra/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+23
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=0.5
      df = pd.concat((df,small_df),axis=0,sort=True)
  '''
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
  iao=np.load('../../../pyscf/ub3lyp_full/b3lyp_iao_b.pickle')
  iao=iao[:,act_iao]
  
  #LOAD IN MOS
  act_mo=[5,6,7,8,9,10,11,12,13]
  chkfile='../../../pyscf/chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  mo=m.mo_coeff[:,act_mo]
  s=m.get_ovlp()

  #IAO ordering: ['del','del','yz','xz','x','y','z2','z','s']
  #MO ordering:  dxz, dyz, dz2, delta, delta, px, py, pz, 4s

  df['iao_n_3dd']=0
  df['iao_n_3dpi']=0
  df['iao_n_3dz2']=0
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

    df['iao_n_3dd'].iloc[z]=np.sum(np.diag(e)[[0,1]])
    df['iao_n_3dpi'].iloc[z]=np.sum(np.diag(e)[[2,3]])
    df['iao_n_3dz2'].iloc[z]=np.diag(e)[6]
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
#LOG METHODS
#Main loop for log regression
def main_log(df,model_list,betas=np.arange(0,3.75,0.25)):
  ncv=5
  zz=0
  regr_df=None
  oneparm_df=None
  ed_df=None
  for model in model_list:
    print("model =============================================== "+str(zz))
    for beta in betas:
      print("beta -------------------------------------------------- "+str(beta))
      weights=np.exp(-beta*(df['energy']-min(df['energy'])))
      X=df[model+['energy']]
      X=sm.add_constant(X)
      X['weights']=weights

      exp_parms_list, d0 = regr_log(X,model)
      d1 = oneparm_valid_log(X,ncv,model)
      d2 = ed_log(model,exp_parms_list)
    
      d0['beta']=beta
      d0['model']=zz
      d1['beta']=beta
      d1['model']=zz
      d2['beta']=beta
      d2['model']=zz
 
      if(regr_df is None): regr_df = d0
      else: regr_df = pd.concat((regr_df, d0),axis=0)
      if(oneparm_df is None): oneparm_df = d1
      else: oneparm_df = pd.concat((oneparm_df, d1),axis=0)
      if(ed_df is None): ed_df = d2
      else: ed_df = pd.concat((ed_df, d2),axis=0)
    zz+=1
  return regr_df, oneparm_df, ed_df

#Log cost function regression
def regr_log(X,model):
  print("REGR LOG ~~~~~~~~~~~~~~~~~")
  exp_parms_list, yhat, yerr_u, yerr_l = log_fit_bootstrap(X,n=50)

  param_names=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz',
  'mo_t_ds','mo_t_sz','Jsd','Us']
  params=[]
  params_u=[]
  params_l=[]

  exp_mu = np.mean(exp_parms_list,axis=0)
  l = exp_mu - np.percentile(exp_parms_list,2.5,axis=0)
  u = np.percentile(exp_parms_list,97.5, axis=0) - exp_mu

  for parm in param_names:
    if(parm in model): 
      params.append(exp_mu[model.index(parm)+1])
      params_u.append(u[model.index(parm)+1])
      params_l.append(l[model.index(parm)+1])
    else: 
      params.append(0) 
      params_u.append(0)
      params_l.append(0)
  d = pd.DataFrame(data=np.array(params + params_u + params_l)[:,np.newaxis].T, columns = param_names + 
  [x+'_u' for x in param_names] + [x+'_l' for x in param_names],index=[0])
  return exp_parms_list, d

#Single parameter goodness of fit validation
def oneparm_valid_log(X,ncv,model):
  print("ONEPARM LOG ~~~~~~~~~~~~~~~~~")
  #Figure out which parameters are in my list
  param_names=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz',
  'mo_t_ds','mo_t_sz','Jsd','Us']
  params=[]
  for parm in param_names:
    if(parm in model): params.append(1)
    else: params.append(0)
 
  kf=KFold(n_splits=ncv,shuffle=True)
  rmsebar_train=[]
  rmsebar_test=[]
  r2_train=[]
  r2_test=[]
  rmse_train=[]
  rmse_test=[]
  evals=[]
  for train_index,test_index in kf.split(X):
    X_train,X_test=X.iloc[train_index],X.iloc[test_index]
    exp_parms, yhat, yerr_u, yerr_l = log_fit_bootstrap(X_train,n=50)
    exp_parms = np.mean(exp_parms,axis=0)

    y_train = X_train['energy']
    y_test = X_test['energy']
    w_train = X_train['weights']
    w_test = X_test['weights']
    yhat_train = np.dot(exp_parms,X_train.drop(columns=['energy','weights']).values.T)
    yhat_test = np.dot(exp_parms,X_test.drop(columns=['energy','weights']).values.T)

    rmsebar_train.append(rmse_bar(yhat_train,y_train,w_train))
    rmsebar_test.append(rmse_bar(yhat_test,y_test,w_test))
    r2_train.append(r2_score(yhat_train,y_train,w_train))
    r2_test.append(r2_score(yhat_test,y_test,w_test))
    rmse_train.append(mean_squared_error(y_train,yhat_train,w_train))
    rmse_test.append(mean_squared_error(y_test,yhat_test,w_test))

  rmsebar_train=np.array(rmsebar_train)
  rmsebar_test=np.array(rmsebar_test)
  r2_train=np.array(r2_train)
  r2_test=np.array(r2_test)
  rmse_train=np.array(rmse_train)
  rmse_test=np.array(rmse_test)

  param_names=['RMSEbarcv_mu_train','RMSEbarcv_std_train','RMSEbarcv_mu_test','RMSEbarcv_std_test',
  'R2cv_mu_train','R2cv_std_train','R2cv_mu_test','R2cv_std_test',
  'RMSEcv_mu_train','RMSEcv_std_train','RMSEcv_mu_test','RMSEcv_std_test']
  params=[np.mean(rmsebar_train),np.std(rmsebar_train),np.mean(rmsebar_test),np.std(rmsebar_test),
  np.mean(r2_train),np.std(r2_train),np.mean(r2_test),np.std(r2_test),
  np.mean(rmse_train),np.std(rmse_train),np.mean(rmse_test),np.std(rmse_test)]

  d=pd.DataFrame(data=np.array(params)[np.newaxis,:],columns=param_names,index=[0])
  return d

#Exact diagonalization using log regression
def ed_log(model,exp_parms_list):
  print("ED LOG ~~~~~~~~~~~~~~~~~")
  full_df=None
  zz=-1
  for exp_parms in exp_parms_list:
    zz+=1
    #Figure out which parameters are in my list
    param_names=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz',
    'mo_t_ds','mo_t_sz','Jsd','Us']
    params=[]
    for parm in param_names:
      if(parm in model): params.append(exp_parms[model.index(parm)+1])
      else: params.append(0)
    
    norb=9
    nelec=(8,7)
    nroots=30
    res1=ED(params,nroots,norb,nelec)

    nelec=(9,6)
    nroots=30
    res3=ED(params,nroots,norb,nelec)
  
    E = res1[0]
    Sz = np.ones(len(E))*0.5
    '''
    dm = res1[2] + res1[3]
    n_3d = dm[:,0,0]+dm[:,1,1]+dm[:,2,2]+dm[:,3,3]+dm[:,6,6]
    n_2ppi = dm[:,4,4]+dm[:,5,5]
    n_2pz = dm[:,7,7]
    n_4s = dm[:,8,8]
    t_pi = 2*(dm[:,3,4]+dm[:,2,5])
    t_ds = 2*dm[:,6,8]
    t_dz = 2*dm[:,6,7]
    t_sz = 2*dm[:,7,8]
    d = pd.dataframe({'energy':e,'sz':sz,'iao_n_3d':n_3d,'iao_n_2pz':n_2pz,'iao_n_2ppi':n_2ppi,'iao_n_4s':n_4s,
    'iao_t_pi':t_pi,'iao_t_ds':t_ds,'iao_t_dz':t_dz,'iao_t_sz':t_sz,'iao_us':res1[4],'iao_jsd':res1[5]})
    '''
    d = pd.DataFrame({'energy':E,'Sz':Sz})
    ci=np.array(res1[1])
    ci=np.reshape(ci,(ci.shape[0],ci.shape[1]*ci.shape[2]))
    d['ci']=list(ci)
  
    E = res3[0]
    Sz = np.ones(len(E))*1.5
    '''
    dm = res3[2] + res3[3]
    n_3d = dm[:,0,0]+dm[:,1,1]+dm[:,2,2]+dm[:,3,3]+dm[:,6,6]
    n_2ppi = dm[:,4,4]+dm[:,5,5]
    n_2pz = dm[:,7,7]
    n_4s = dm[:,8,8]
    t_pi = 2*(dm[:,3,4]+dm[:,2,5])
    t_ds = 2*dm[:,6,8]
    t_dz = 2*dm[:,6,7]
    t_sz = 2*dm[:,7,8]
    d2 =pd.DataFrame({'energy':E,'Sz':Sz,'iao_n_3d':n_3d,'iao_n_2pz':n_2pz,'iao_n_2ppi':n_2ppi,'iao_n_4s':n_4s,
    'iao_t_pi':t_pi,'iao_t_ds':t_ds,'iao_t_dz':t_dz,'iao_t_sz':t_sz,'iao_Us':res3[4],'iao_Jsd':res3[5]})
    '''
    d2 = pd.DataFrame({'energy':E,'Sz':Sz})
    ci=np.array(res3[1])
    ci=np.reshape(ci,(ci.shape[0],ci.shape[1]*ci.shape[2]))
    d2['ci']=list(ci)

    d=pd.concat((d,d2),axis=0)
    d['energy']-=min(d['energy'])
    d['bs_index']=zz

    if(full_df is None): full_df = d
    else: full_df = pd.concat((full_df,d),axis=0)
  return full_df











######################################################################################
#LOG METHODS
#CIs and means for ED
def av_ed_log(eig_df,has_eig=False):
  av_df = None
  for model in range(max(eig_df['model'])+1):
    for beta in [0.0,1.0,2.0]:
      for eig in range(max(eig_df['eig'])+1):
        sub_df = eig_df[(eig_df['model']==model) & (eig_df['eig']==eig) &(eig_df['beta']==beta)]
        data = sub_df.values
        means = np.mean(data,axis=0)
        u = np.percentile(data,97.5,axis=0) - means
        l = means - np.percentile(data,2.5,axis=0)

        d=pd.DataFrame(data=np.array(list(means) + list(u)+list(l))[:,np.newaxis].T,
        columns=list(sub_df) + [x+'_u' for x in list(sub_df)] + [x+'_l' for x in list(sub_df)])

        if(av_df is None): av_df = d
        else: av_df = pd.concat((av_df,d),axis=0)
  return av_df

def sort_eigs(eig_df):
  '''
  #Separate by spin or else they'll mix
  d['eig']=-1
  add=0
  for Sz in [0.5,1.5]:
    s=d[d['Sz']==Sz].drop(columns=['eig','Sz']).values[:,:]
    o=full_df[full_df['Sz']==Sz].drop(columns=['eig','Sz']).values[:s.shape[0],:]
    o=preprocessing.scale(o) #Scaled unit variance and zero mean, normalized vector space
    s=preprocessing.scale(s)

    cost = scipy.spatial.distance.cdist(o,s)
    assert(cost.shape[0]==s.shape[0])
    assert(cost.shape[1]==s.shape[0])
    row_ind, col_ind = linear_sum_assignment(cost)
    
    d['eig'][d['Sz']==Sz] = col_ind + add
    add += s.shape[0]
  '''
  return eig_df

######################################################################################
#LOG METHODS (PLOTS) 
#Plot regression parameters
def plot_regr_log(save=False):
  full_df=pd.read_pickle('analysis/regr_log.pickle')

  for model in [0,1,4,8,3,7]:
    s = full_df[full_df['model']==model]
    print(s.iloc[0])
  #s = full_df[full_df['mo_t_pi']!=0]
  #print(set(s['model']))
  exit(0)

  #for model in [0]:
  #  print(full_df[(full_df['model']==model)*(full_df['beta']==2.0)].iloc[0])
  #exit(0)

  model=[]
  for i in range(16):
    model+=list(np.linspace(i,i+0.75,15))
  full_df['model']=model

  for parm in list(full_df)[:9]:
    g = sns.FacetGrid(full_df,hue='beta')
    g.map(plt.errorbar, "model", parm, parm+'_err',fmt='.').add_legend()
    plt.xlabel('Model, eV')
    plt.ylabel(parm+', eV')
    if(save): plt.savefig('analysis/regr_'+parm+'.pdf',bbox_inches='tight')
    else: plt.show()
    plt.close()

#Plot single parameter goodness of fit 
def plot_oneparm_valid_log(save=False):
  full_df=pd.read_pickle('analysis/oneparm_log.pickle')
  model=[]
  for i in range(32):
    model+=list(np.linspace(i,i+0.75,15))
  full_df['model']=model

  #full_df = full_df[full_df['beta']==1.5]

  g = sns.FacetGrid(full_df,hue='beta')
  g.map(plt.errorbar, "model", "R2cv_mu_train", "R2cv_std_train",fmt='.').add_legend()
  plt.xlabel('Model, eV')
  plt.ylabel('R2cv_train, eV')
  if(save): plt.savefig('analysis/oneparm_train_r2.pdf',bbox_inches='tight')
  else: plt.show()
  plt.close()

  g = sns.FacetGrid(full_df,hue='beta')
  g.map(plt.errorbar, "model", "R2cv_mu_test", "R2cv_std_test",fmt='.').add_legend()
  plt.xlabel('Model, eV')
  plt.ylabel('R2cv_test, eV')
  if(save): plt.savefig('analysis/oneparm_test_r2.pdf',bbox_inches='tight')
  else: plt.show()
  plt.close()
  
  g = sns.FacetGrid(full_df,hue='beta')
  g.map(plt.errorbar, "model", "RMSEcv_mu_train", "RMSEcv_std_train",fmt='.').add_legend()
  plt.xlabel('Model, eV')
  plt.ylabel('RMSEcv_train, eV')
  if(save): plt.savefig('analysis/oneparm_train_rmse.pdf',bbox_inches='tight')
  else: plt.show()
  plt.close()

  g = sns.FacetGrid(full_df,hue='beta')
  g.map(plt.errorbar, "model", "RMSEcv_mu_test", "RMSEcv_std_test",fmt='.').add_legend()
  plt.xlabel('Model, eV')
  plt.ylabel('RMSEcv_test, eV')
  if(save): plt.savefig('analysis/oneparm_test_rmse.pdf',bbox_inches='tight')
  else: plt.show()
  plt.close()
 
  return -1 

#Plot eigenvalues and eigenproperties
def plot_ed_log(full_df,save=True):
  norm = mpl.colors.Normalize(vmin=0, vmax=3.75)
  #FULL EIGENPROPERTIES and EIGENVALUES
  
  av_df = pd.read_pickle('analysis/av_sorted_ed_log_og_d.pickle')
  #av_df = pd.read_pickle('analysis/av_ed_log_d.pickle')

  ## TO GET NICE FORMATTING
  g = sns.FacetGrid(av_df,col='model',col_wrap=3,hue='Sz')
  limits = [(0.5,2.5),(2.5,4.5),(1.5,4.5),(0.5,2.5),(1.5,4.5),(0,1.5),
  (-1.0,1.5),(-1.5,1.5),(-1.5,1.5),(-1.5,1.5),(-1,1),(-0.5,1.0)]
  for model in np.arange(32):
    for beta in [0.0,1.0,2.0]:
      rgba_color = cm.Blues(norm(3.75-beta))
      rgba_color2 = cm.Oranges(norm(3.75-beta))
      z=-1
      fig, axes = plt.subplots(nrows=2,ncols=6,sharey=True,figsize=(12,6))
      for parm in ['iao_n_3dz2','iao_n_3dpi','iao_n_3dd','iao_n_2pz','iao_n_2ppi','iao_n_4s',
      'iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz','iao_Jsd','iao_Us']:
        z+=1
        ax = axes[z//6,z%6]

        p=parm
        if(parm=='iao_Jsd'): p = 'Jsd'
        if(parm=='iao_Us'):  p = 'Us'
        
        full_df['energy']-=min(full_df['energy'])
        full_df = full_df[full_df['basestate']==-1]
    
        f_df = full_df[full_df['Sz']==0.5]
        x = f_df[p].values
        y = f_df['energy'].values
        yerr = f_df['energy_err'].values
        ax.errorbar(x,y,yerr,fmt='s',c=rgba_color,alpha=0.5)

        f_df = full_df[full_df['Sz']==1.5]
        x = f_df[p].values
        y = f_df['energy'].values
        yerr = f_df['energy_err'].values
        ax.errorbar(x,y,yerr,fmt='s',c=rgba_color2,alpha=0.5)

        sub_df = av_df[(av_df['model']==model)&(av_df['Sz']==0.5)&(av_df['beta']==beta)]
        x=sub_df[parm].values
        xerr_u=sub_df[parm+'_u'].values
        xerr_d=sub_df[parm+'_l'].values
        y=sub_df['energy'].values
        yerr_u=sub_df['energy_u'].values
        yerr_d=sub_df['energy_l'].values
        ax.errorbar(x,y,xerr=[xerr_d,xerr_u],yerr=[yerr_d,yerr_u],markeredgecolor='k',fmt='o',c=rgba_color)
       
        sub_df = av_df[(av_df['model']==model)&(av_df['Sz']==1.5)&(av_df['beta']==beta)]
        x=sub_df[parm].values
        xerr_u=sub_df[parm+'_u'].values
        xerr_d=sub_df[parm+'_l'].values 
        y=sub_df['energy'].values
        yerr_u=sub_df['energy_u'].values
        yerr_d=sub_df['energy_l'].values
        ax.errorbar(x,y,xerr=[xerr_d,xerr_u],yerr=[yerr_d,yerr_u],markeredgecolor='k',fmt='o',c=rgba_color2)
        
        ax.set_ylim((-0.2,6.0))
        ax.set_xlabel(parm)
        ax.set_ylabel('energy (eV)')
        ax.set_xlim(limits[z])
      if(save): plt.savefig('analysis/sorted_ed_'+str(model)+'_'+str(beta)+'_log_og.pdf',bbox_inches='tight')
      else: plt.show()
      plt.clf()
  
  '''
  fig, axes = plt.subplots(nrows=2,ncols=6,sharey=True,figsize=(12,6))
  for model in [2,13,14,12,25,5,29,0,1,3,9,22]:
    rgba_color = cm.Blues(norm(3.75-beta))
    rgba_color2 = cm.Oranges(norm(3.75-beta))
    z=-1
    for parm in ['iao_n_3dz2','iao_n_3dpi','iao_n_3dd','iao_n_2pz','iao_n_2ppi','iao_n_4s',
    'iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz','iao_Jsd','iao_Us']:
      z+=1
      ax = axes[z//6,z%6]

      sub_df = av_df[(av_df['model']==model)&(av_df['Sz']==0.5)]
      x=sub_df[parm].values
      xerr=sub_df[parm+'_err'].values
      y=sub_df['energy'].values
      yerr=sub_df['energy_err'].values
      ax.errorbar(x,y,xerr=xerr,yerr=yerr,markeredgecolor='k',fmt='o',c=rgba_color)
     
      sub_df = av_df[(av_df['model']==model)&(av_df['Sz']==1.5)]
      x=sub_df[parm].values
      xerr=sub_df[parm+'_err'].values
      y=sub_df['energy'].values
      yerr=sub_df['energy_err'].values
      ax.errorbar(x,y,xerr=xerr,yerr=yerr,markeredgecolor='k',fmt='o',c=rgba_color2)
      
      ax.set_ylim((-0.2,2.5))
      ax.set_xlabel(parm)
      ax.set_ylabel('energy (eV)')
  plt.show()
  plt.clf()
  '''

def plot_fit_log(X,save=True,fname=None):
  print("PLOT FIT ~~~~~~~~~~~~~~~~~")
  exp_parms_list, yhat, yerr_u, yerr_l = log_fit_bootstrap(X,n=20)
  print(np.around(exp_parms_list.mean(axis=0),2))
  print(np.around(exp_parms_list.std(axis=0),2))

  X['pred']=yhat
  X['pred_err']=(yerr_u - yerr_l)/2

  g = sns.FacetGrid(X,hue='Sz')
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
  plt.plot(X['energy'],X['energy'],'k--')
  plt.title('Regression, 95% CI')
  plt.xlabel('Predicted Energy, eV')
  plt.ylabel('DMC Energy, eV')
  if(save): plt.savefig(fname+'.pdf',bbox_inches='tight')
  else: plt.show()
  plt.close()
  X=X[X['basestate']==-1]
  g = sns.FacetGrid(X,hue='Sz')
  g.map(plt.errorbar, "pred", "energy", "energy_err","pred_err",fmt='o').add_legend()
  plt.plot(X['energy'],X['energy'],'k--')
  plt.title('Regression base states only, 95% CI')
  plt.xlabel('Predicted Energy, eV')
  plt.ylabel('DMC Energy, eV')
  if(save): plt.savefig(fname+'_base.pdf',bbox_inches='tight')
  else: plt.show()
  plt.close()  

  return exp_parms_list, yhat, yerr_u, yerr_l

def calc_density(df,dE):
  density=np.zeros(df.shape[0])
  df['energy']=np.around(df['energy'],3)
  for p in range(df.shape[0]):
    density[p] = np.sum((df['energy']<=(df['energy'].iloc[p] + dE))&(df['energy']>=(df['energy'].iloc[p] - dE)))
  df['density']=density

  return df

def plot_noiser2(save=False):
  ed_df=pd.read_pickle('analysis/av_ed_log.pickle')
  r2_df=pd.read_pickle('analysis/oneparm_log.pickle')

  betas=[]
  models=[]
  e_errs=[]
  p_errs=[]
  r2s=[]
  r2_errs=[]
  rmsebars=[]
  rmsebar_errs=[]
  for beta in np.arange(0,3.75,0.25):
    for model in range(32):
      a=ed_df[(ed_df['beta']==beta)&(ed_df['model']==model)]
      a=calc_density(a)
      b=r2_df[(r2_df['beta']==beta)&(r2_df['model']==model)]
      betas.append(beta)
      models.append(model)
      r2s.append(b['R2cv_mu_test'])
      r2_errs.append(b['R2cv_std_test'])
      rmsebars.append(b['RMSEbarcv_mu_test'])
      rmsebar_errs.append(b['RMSEbarcv_std_test'])
      e_errs.append(np.sum(a['energy_err'].values/a['density'].values))
      p_err=0
      for col in ['iao_n_3d', 'iao_n_2pz', 'iao_n_2ppi', 'iao_n_4s',
      'iao_t_pi', 'iao_t_ds', 'iao_t_dz', 'iao_t_sz', 'iao_Us', 'iao_Jsd']:
        p_err += np.sum(a[col+'_err'].values/a['density'].values)
      p_errs.append(p_err)

  ret_df = pd.DataFrame(columns=['beta','model','R2cv_mu_test','R2cv_std_test','RMSEbarcv_mu_test','RMSEbarcv_std_test',
  'e_err','p_err'],
  data=np.array([betas,models,r2s,r2_errs,rmsebars,rmsebar_errs,e_errs,p_errs]).T)

  markers=['.','o','v','^','<','>','8','s','P','p','h','H','*','X','D','d']*2
  gs = mpl.gridspec.GridSpec(1, 1)
  ax1 = plt.subplot(gs[0])
  for beta in [2.0]:
    for model in range(32):
      pdf=ret_df[(ret_df['model']==model)&(ret_df['beta']==beta)]
      if(model < 16):
        ax1.errorbar(pdf['e_err'].values,pdf['p_err'].values,c='k',marker=markers[model],
        label='model '+str(model))
      else: 
        ax1.errorbar(pdf['e_err'].values,pdf['p_err'].values,c='r',marker=markers[model],
        label='model '+str(model))
  '''
  for model in np.arange(16):
    pdf=ret_df[(ret_df['model']==model)]
    ax1.plot(pdf[error].values,pdf['RMSEbarcv_mu_test'].values,c='gray',ls='--')
  '''

  #cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=cmap, norm=norm,orientation='vertical')
  ax1.legend(loc='best')
  ax1.set_xlabel('e_err')
  ax1.set_ylabel('p_err')
  if(save): plt.savefig('analysis/evsp_err.pdf',bbox_inches='tight')
  else: plt.show()
  plt.clf()

def plot_Xerr(df,save=True):
  ed_df=pd.read_pickle('analysis/av_ed_log.pickle')
  df['iao_Jsd']=df['Jsd']
  df['iao_Us']=df['Us']
  var=['iao_n_3d','iao_n_2pz','iao_n_2ppi','iao_n_4s','iao_t_pi','iao_t_dz','iao_t_ds','iao_t_sz','iao_Jsd','iao_Us']
  fig, ax = plt.subplots(2,5,sharey=True)
  beta=2.0
  z=0
  for parm in var:
    zb=df[parm]
    for model in range(16):
      za=ed_df[(ed_df['model']==model)*(ed_df['beta']==beta)]
      #za = za.iloc[[0,1,2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29]]
      
      lo = (za[parm] - za[parm+'_err']).min()
      hi = (za[parm] + za[parm+'_err']).max()

      ax[z//5,z%5].plot([lo,hi],[model+1,model+1],marker='|',c='cornflowerblue')
    
    #MIN MAX
    ax[z//5,z%5].axvline(zb.min(),c='k',ls='--')
    ax[z//5,z%5].axvline(zb.max(),c='k',ls='--')
    
    #HISTOGRAM
    bins, edges = np.histogram(zb,bins=10,density=True)
    bins*=16/max(bins)
    ax[z//5,z%5].bar(x=edges[:-1],height=bins,width=edges[1:] - edges[:-1],color='gray',alpha=0.5)
    ax[z//5,z%5].set_xlabel(parm)
    z+=1
  plt.show()

def compare_spectrum():
  beta = 2.0
  bs_index = []
  for i in range(100): 
    bs_index += [i]*40 
  bs_index*=15*32
  
  df = pd.read_pickle('analysis/ed_log.pickle')
  df['bs_index'] = bs_index
  df=df[df['beta']==beta]
  models = [0,1,3,4,7,8]
  z=(df['model']==0)
  for i in models:
    z+=(df['model']==i)
  df=df[z]
  
  g=sns.FacetGrid(df, col='model', col_wrap = 3, hue='Sz')
  g=g.map(plt.errorbar,"index","energy",fmt='.')
  plt.show()

######################################################################################
#RUN
import sklearn 

def analyze(df=None,save=False):
  #LOG DATA COLLECTION
  '''model = ['mo_n_4s','mo_n_2ppi','mo_t_dz','mo_t_sz','mo_t_ds','Us']
  X = df[model]
  X = sm.add_constant(X)
  y = df['energy']
  ols = sm.OLS(y,X).fit()
  plt.plot(ols.predict(),y,'ob')
  plt.plot(y,y,'--g')
  plt.show()
  
  print(ols.summary())
  exit(0)
  #eig_df = ed_log(model,[ols.params])
  #print(eig_df)
  '''

  model = ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz','mo_t_ds','mo_t_sz','Jsd','Us']
  X = df[model]
  y = df['energy']
  '''    
  X = sm.add_constant(X)
  ols = sm.OLS(y,X).fit()
  print(ols.summary())
  exit(0)
  '''

  nz = []
  r2 = []
  r2_err = []
  coeff = []
  coeff_err = []
  for alpha in np.arange(1e-6,0.105,0.002):
    lasso = sklearn.linear_model.Lasso(alpha=alpha,fit_intercept=True,selection='random')
    lasso = lasso.fit(X,y)
    ind = lasso.coef_.nonzero()[0]
    selected_model = np.array(model)[ind]
  
    r2_bs = []
    coeff_bs = []
    for n in range(20):
      data = df 
      data = df.sample(n=df.shape[0],replace=True)
      Xp = data[selected_model]
      #Xp = sm.add_constant(Xp)
      Xp['const'] = 1
      yp = data['energy']
      
      ols = sm.OLS(yp,Xp).fit()
      #if(n==0): print(ols.summary())
      r2_bs.append(r2_score(ols.predict(),yp))
      coeff_bs.append(ols.params)
    r2_bs = np.array(r2_bs)
    r2.append(r2_bs.mean(axis=0))
    r2_err.append(r2_bs.std(axis=0))
    
    coeff_bs = np.array(coeff_bs)
    coeff = coeff_bs.mean(axis=0)
    coeff_err = coeff_bs.std(axis=0)

    nz.append(len(selected_model))
    print(selected_model)
    print(coeff)
    print(coeff_err)

  plt.errorbar(nz,r2,yerr=r2_err,fmt='o')
  plt.show()
  exit(0)

  #LASSO TELLS US THAT A 5 PARAMETER MODEL IS THE BEST OUT OF THESE
  #THE SELECTED 5 PARAMETER MODEL IS ALWAYS THE SAME!

  '''
  df = df[df['basestate']==-1]
  df['energy']/=27.2114
  df = df.sort_values(by='energy')
  print(df[['energy','Sz','mo_n_3dz2','mo_n_3dpi','mo_n_2ppi','mo_n_2pz','mo_n_4s','mo_t_pi','mo_t_dz','mo_t_ds','mo_t_sz','Us','Jsd']])
  exit(0)

  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_n_3dpi','mo_n_3dz2','mo_t_pi','mo_t_ds','mo_t_sz','mo_t_dz','Us','Jsd']
  X=df[['energy','energy_err','Sz','basestate']+model]
  X=sm.add_constant(X)
  X['weights']=1
  parms, __ , __, __ =plot_fit_log(X,save=False)
  print(model)
  ed_df = ed_log(model,parms)
  
  y = ed_df['energy']
  plt.plot(np.ones(len(y[ed_df['Sz']==0.5])),y[ed_df['Sz']==0.5],'o')
  plt.plot(np.ones(len(y[ed_df['Sz']==1.5]))+1,y[ed_df['Sz']==1.5],'o')
  plt.show()
  exit(0)
  '''

  '''
  #Generate all possible models
  X=df[['mo_n_4s','mo_n_2ppi','mo_n_2pz','Us']]
  hopping=df[['Jsd','mo_t_pi','mo_t_dz','mo_t_ds','mo_t_sz']]
  y=df['energy']
  model_list=[list(X)]
  for n in range(1,hopping.shape[1]+1):
    s=set(list(hopping))
    models=list(map(set,itertools.combinations(s,n)))
    model_list+=[list(X)+list(m) for m in models]
  print(len(model_list))

  df0,df1,df2=main_log(df,model_list,betas=[0.0,1.0,2.0])
  print(df0)
  print(df1)
  print(df2)
  df0.to_pickle('analysis/regr_log_og.pickle')
  df1.to_pickle('analysis/oneparm_log_og.pickle')
  df2.to_pickle('analysis/ed_log_og.pickle')
  exit(0)
  '''

  #plot_oneparm_valid_log(save=False)
  #exit(0)
  
  '''
  #Sort/group eigenvalues
  df = pd.read_pickle('analysis/ed_log_og.pickle')
  #df = df[df['beta']==0.0]
  df3 = None
  for model in range(32):
    for beta in [0.0,1.0,2.0]:
      df2 = df[(df['model']==model)&(df['beta']==beta)]
      for j in range(max(df2['bs_index'])+1):
        offset = 0
        for Sz in [0.5,1.5]:
          a = df2[(df2['bs_index']==0)&(df2['Sz']==Sz)]#.iloc[:m[Sz]]
          amat = np.array(list(a['ci']))

          b = df2[(df2['bs_index']==j)&(df2['Sz']==Sz)]#.iloc[:m[Sz]]
          bmat = np.array(list(b['ci']))
      
          #Apply permutation to pair up non degenerate states
          cost = -1.*np.dot(amat,bmat.T)**2
          row_ind, col_ind = linear_sum_assignment(cost)
          bmat = bmat[col_ind,:]
          
          #Gotta do some extra work for the degenerate states
          #Get connected groups
          abdot = np.dot(amat,bmat.T)
          mask = (abdot**2 > 1e-5)
          connected_sets, home_nodes = find_connected_sets(mask)
         
          #Loop connected groups to see which ones correspond to degenerate states
          for i in connected_sets:
            len_check = False
            sub_ind = None
            sum_check = False
            eig_check = False
            #Check length as criteria for degeneracy
            if(len(connected_sets[i])>1):
              sub_ind=list(connected_sets[i])
              len_check = True

            if(len_check):
              degen_a = len(set(np.around(a['energy'].iloc[row_ind[sub_ind]],6)))
              degen_b = len(set(np.around(b['energy'].iloc[col_ind[sub_ind]],6)))
              if((degen_a == 1)&(degen_b ==1)): eig_check = True

            #Check that the degenerate space is actually spanned properly
            if(eig_check):
              sub_mat = abdot[sub_ind][:,sub_ind]
              bmat[row_ind[sub_ind],:]=np.dot(sub_mat,bmat[row_ind[sub_ind],:])
          
          #We finally have bmat ordered correctly and everything 
          dtmp = pd.DataFrame({'energy':b['energy'].values[col_ind],'Sz':b['Sz'].values,'bs_index':b['bs_index'].values})
          dtmp['ci']=list(bmat)
          dtmp['eig']=np.arange(dtmp.shape[0]) + offset
          dtmp['model']=b['model'].values
          dtmp['beta']=beta
          offset += dtmp.shape[0]
          if(df3 is None): df3 = dtmp
          else: df3 = pd.concat((df3,dtmp),axis=0)
  df3.to_pickle('analysis/sorted_ed_log_og.pickle')

  #Calculate descriptors after sorting
  from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
  import scipy as sp 

  df3 = pd.read_pickle('analysis/sorted_ed_log_og.pickle')
  df4 = None
  mol = gto.Mole()
  cis = fci.direct_uhf.FCISolver(mol)

  sigUs = []
  sigJsd = []
  sigNdz2 = []
  sigNdpi = []
  sigNdd =[]
  sigNd = []
  sigN2pz = []
  sigN2ppi = []
  sigN4s = []
  sigTpi = []
  sigTds = []
  sigTdz = []
  sigTsz = []

  for i in range(df3.shape[0]):
    ci = df3['ci'].iloc[i]
    norb=9
    nelec=(8,7)
    if(df3['Sz'].iloc[i]==1.5): nelec=(9,6)   
    ci = ci.reshape((sp.misc.comb(norb,nelec[0],exact=True),sp.misc.comb(norb,nelec[1],exact=True)))
    dm2=cis.make_rdm12s(ci,norb,nelec)
    
    #Parameters
    sigUs.append(dm2[1][1][8,8,8,8])
    
    Jsd = 0
    for i in [0,1,2,3,6]:
      Jsd += 0.25*(dm2[1][0][8,8,i,i] + dm2[1][2][8,8,i,i] - dm2[1][1][8,8,i,i] - dm2[1][1][i,i,8,8])-\
             0.5*(dm2[1][1][i,8,8,i] + dm2[1][1][8,i,i,8]) 
    sigJsd.append(Jsd)
  
    dm = dm2[0][0] + dm2[0][1]
    
    sigNdz2.append(dm[6,6])
    sigNdpi.append(dm[2,2]+dm[3,3])
    sigNdd.append(dm[0,0]+dm[1,1])
    sigNd.append(dm[0,0]+dm[1,1]+dm[2,2]+dm[3,3]+dm[6,6])
    sigN2pz.append(dm[7,7])
    sigN2ppi.append(dm[4,4]+dm[5,5])
    sigN4s.append(dm[8,8])
    sigTpi.append(2*(dm[3,4]+dm[2,5]))
    sigTds.append(2*dm[6,8])
    sigTdz.append(2*dm[6,7])
    sigTsz.append(2*dm[7,8])
  
  df3['iao_n_3dz2']=sigNdz2
  df3['iao_n_3dpi']=sigNdpi
  df3['iao_n_3dd']=sigNdd
  df3['iao_n_3d']=sigNd
  df3['iao_n_2pz']=sigN2pz
  df3['iao_n_2ppi']=sigN2ppi
  df3['iao_n_4s']=sigN4s
  df3['iao_t_pi']=sigTpi
  df3['iao_t_ds']=sigTds
  df3['iao_t_dz']=sigTdz
  df3['iao_t_sz']=sigTsz
  df3['iao_Us']=sigUs
  df3['iao_Jsd']=sigJsd

  df3.to_pickle('analysis/sorted_ed_log_og_d.pickle')
  print(df3)
  '''

  #Average everything
  #df3 = pd.read_pickle('analysis/sorted_ed_log_og_d.pickle')
  #av_df3 = av_ed_log(df3.drop(columns=['ci']))
  #av_df3.to_pickle('analysis/av_sorted_ed_log_og_d.pickle')
  #exit(0)

  plot_ed_log(df)
  exit(0)
  
  '''
  from sklearn.decomposition import PCA 
  pca = PCA(n_components=3)
  X = df[['mo_n_2ppi','mo_n_3dpi','mo_t_pi']]
  pca.fit(X)
  print(pca.explained_variance_ratio_)
  X=np.dot(X,pca.components_.T)
  df[['mo_n_2ppi','mo_n_3dpi','mo_t_pi']]=X

  pca = PCA(n_components=6)
  X = df[['mo_n_2pz','mo_n_3dz2','mo_n_4s','mo_t_ds','mo_t_dz','mo_t_sz']]
  pca.fit(X)
  print(pca.explained_variance_ratio_)
  X=np.dot(X,pca.components_.T)
  df[['mo_n_2pz','mo_n_3dz2','mo_n_4s','mo_t_ds','mo_t_dz','mo_t_sz']]=X

  model = ['mo_n_2ppi','mo_n_3dpi','mo_t_pi','mo_n_2pz','mo_n_3dz2','mo_n_4s','mo_t_ds','mo_t_sz','Us']
  X = df[model]
  X = sm.add_constant(X)
  y = df['energy'].values
  ols = sm.OLS(y,X).fit()
  print(ols.summary())
  
  plt.plot(y,y,'--g')
  plt.plot(ols.predict(),y,'ob')
  plt.show()

  ind = df['basestate']==-1
  plt.plot(y[ind],y[ind],'--g')
  plt.plot(ols.predict(X)[ind],y[ind],'ob')
  plt.show()
  '''
  '''
  y = df['energy']
  y -= min(y)
  vars = []
  for cut in np.arange(0.25,4.25,0.25):
    vars.append(y[(y<=cut)&(y>=cut-0.25)].var())
  print(np.sum(vars))
  plt.plot(np.arange(0.25,4.24,0.25),vars,'o')
  plt.show()
  '''

  '''
  X = df[['energy','energy_err','Sz','basestate']+model]
  X = sm.add_constant(X)
  X['weights'] = np.ones(X.shape[0])
  plot_fit_log(X,save=False)
  exit(0)
  '''

  #Low noise filter
  '''
  av_df = pd.read_pickle('analysis/av_sorted_ed_log_d.pickle')
  df['energy']-=min(df['energy'])

  av_df = av_df[av_df['model']==4]
  av_df = av_df[av_df['energy']<=3.0]
  print(av_df[av_df['iao_n_3dpi']<3.5][['energy','iao_n_3dpi','iao_n_2ppi','iao_t_pi','iao_t_dz','iao_t_ds','iao_t_sz']])
  print(av_df[av_df['iao_n_3dd']<3.5][['energy','iao_n_3dd','iao_n_2ppi','iao_t_pi','iao_t_dz','iao_t_ds','iao_t_sz']])
  print(av_df[av_df['iao_t_pi']<0][['energy','iao_n_3dd','iao_n_3dpi','iao_n_3dz2','iao_n_2ppi','iao_n_2pz','iao_n_4s','iao_t_pi','iao_t_dz','iao_t_ds','iao_t_sz']])
  print(av_df[av_df['iao_t_dz']>0][['energy','iao_n_3dd','iao_n_3dpi','iao_n_3dz2','iao_n_2ppi','iao_n_2pz','iao_n_4s','iao_t_pi','iao_t_dz','iao_t_ds','iao_t_sz']])
  print(av_df[av_df['iao_t_sz']>0][['energy','iao_n_3dd','iao_n_3dpi','iao_n_3dz2','iao_n_2ppi','iao_n_2pz','iao_n_4s','iao_t_pi','iao_t_dz','iao_t_ds','iao_t_sz']])
  print(av_df[av_df['iao_t_ds']>0][['energy','iao_n_3dd','iao_n_3dpi','iao_n_3dz2','iao_n_2ppi','iao_n_2pz','iao_n_4s','iao_t_pi','iao_t_dz','iao_t_ds','iao_t_sz']])
  '''

  '''
  av_df['type']='ed'
  df['type']='dmc'
  df['eig']=0
  av_df = pd.concat((av_df,df),axis=0)
  av_df = av_df[av_df['energy']<=3.0]
 
  sns.pairplot(av_df,vars=['energy','eig','iao_n_3dpi','iao_n_3dd','iao_n_2pz','iao_n_2ppi'],hue='type',markers=['o','.'])
  plt.show()
  sns.pairplot(av_df,vars=['energy','eig','iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz'],hue='type',markers=['o','.'])
  plt.show()
  '''
  '''
  models=[0,1,2,3,4,6,7,9,11,12,15]
  for model in models:
    sub_df = av_df[av_df['model']==model]
    err = (sub_df['energy_l'].mean() + sub_df['energy_u'].mean())/2
    print(err)
    if(err <= 0.20):
      #Plot low noise models again, smaller 
      norm = mpl.colors.Normalize(vmin=0, vmax=3.75)
      g = sns.FacetGrid(av_df,col='model',col_wrap=2,hue='Sz')
      limits = [(8.5,10.5),(0.5,2.5),(1.5,4.5),(0,1.5)]
      rgba_color = cm.Blues(norm(3.75-2.0))
      rgba_color2 = cm.Oranges(norm(3.75-2.0))
      z=-1
      fig, axes = plt.subplots(nrows=2,ncols=2,sharey=True,figsize=(3,6))
      for parm in ['iao_n_3d','iao_n_2pz','iao_n_2ppi','iao_n_4s']:
        z+=1
        ax = axes[z//2,z%2]

        #PLOT DMC
        p=parm
        f_df = df[df['Sz']==0.5]
        x = f_df[p].values
        y = f_df['energy'].values
        yerr = f_df['energy_err'].values
        ax.errorbar(x,y,yerr,fmt='.',c=rgba_color,alpha=0.2)
        f_df = df[df['Sz']==1.5]
        x = f_df[p].values
        y = f_df['energy'].values
        yerr = f_df['energy_err'].values
        ax.errorbar(x,y,yerr,fmt='.',c=rgba_color2,alpha=0.2)

        #PLOT ED
        sub_df = av_df[(av_df['model']==model)&(av_df['Sz']==0.5)]
        x=sub_df[parm].values
        xerr_u=sub_df[parm+'_u'].values
        xerr_d=sub_df[parm+'_l'].values
        y=sub_df['energy'].values
        yerr_u=sub_df['energy_u'].values
        yerr_d=sub_df['energy_l'].values
        ax.errorbar(x,y,xerr=[xerr_d,xerr_u],yerr=[yerr_d,yerr_u],markeredgecolor='k',fmt='o',c=rgba_color)
       
        sub_df = av_df[(av_df['model']==model)&(av_df['Sz']==1.5)&(av_df['energy']<=4.5)]
        x=sub_df[parm].values
        xerr_u=sub_df[parm+'_u'].values
        xerr_d=sub_df[parm+'_l'].values 
        y=sub_df['energy'].values
        yerr_u=sub_df['energy_u'].values
        yerr_d=sub_df['energy_l'].values
        ax.errorbar(x,y,xerr=[xerr_d,xerr_u],yerr=[yerr_d,yerr_u],markeredgecolor='k',fmt='o',c=rgba_color2)
        
        ax.set_ylim((-0.2,6.0))
        ax.set_xlabel(parm)
        ax.set_ylabel('energy (eV)')
        ax.set_xlim(limits[z])
      plt.savefig('analysis/small_ed_'+str(model)+'_log.pdf',bbox_inches='tight')
      plt.clf()
  '''
if __name__=='__main__':
  #DATA COLLECTION
  '''
  df=collect_df()
  df=format_df(df)
  df.to_pickle('formatted_gosling.pickle')
  '''
  df = pd.read_pickle('formatted_gosling_og.pickle')
  analyze(df)
