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
#from roks_model import ED_roks
#from uks_model import ED_uks
import itertools
from log import log_fit,log_fit_bootstrap, rmse_bar
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import linear_sum_assignment
import scipy 
pd.options.mode.chained_assignment = None  # default='warn'

######################################################################################
#FROZEN METHODS
#Collect df
def collect_df():
  df=None
  for basestate in range(11,13):
    for gsw in [1.0]:
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/vmc_gosling.pickle' 
      small_df=pd.read_pickle(f)

      small_df['basestate']=basestate
      if(gsw==1.0): small_df['basestate']=-1
      small_df['Sz']=0.5
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0,sort=True)

  for basestate in [6]:
    #for gsw in np.arange(0.1,1.1,0.1):
    for gsw in [1.0]:
      f='../ub3lyp_3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/vmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+13
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
  params_err=[]

  exp_mu = np.mean(exp_parms_list,axis=0)
  l = np.percentile(exp_parms_list,2.5,axis=0)
  u = np.percentile(exp_parms_list,97.5, axis=0)
  exp_err = (u-l)/2

  for parm in param_names:
    if(parm in model): 
      params.append(exp_mu[model.index(parm)+1])
      params_err.append(exp_err[model.index(parm)+1])
    else: 
      params.append(0) 
      params_err.append(0)
  d = pd.DataFrame(data=np.array(params + params_err)[:,np.newaxis].T, columns = param_names + [x+'_err' for x in param_names],index=[0])
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
    for eig in range(max(eig_df['eig'])+1):
      sub_df = eig_df[(eig_df['model']==model) & (eig_df['eig']==eig)]
      data = sub_df.values
      means = np.mean(data,axis=0)
      u = np.percentile(data,97.5,axis=0)
      l = np.percentile(data,2.5,axis=0)
      err = (u - l)/2

      d=pd.DataFrame(data=np.array(list(means) + list(err))[:,np.newaxis].T,
      columns=list(sub_df) + [x+'_err' for x in list(sub_df)])

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

  '''
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
  '''
  
  tpi=[3, 7, 10, 13, 14, 16, 19, 20, 22, 23, 25, 26, 27, 29, 30, 31]
  pi=[]
  for i in range(32): 
    if(i in tpi): pi+=[1]*15
    else: pi+=[0]*15
  full_df['pi']=pi
  g = sns.FacetGrid(full_df,hue='pi')
  g.map(plt.errorbar, "model", "R2cv_mu_test", "R2cv_std_test",fmt='.').add_legend()
  plt.xlabel('Model, eV')
  plt.ylabel('R2cv_test, eV')
  if(save): plt.savefig('analysis/oneparm_test_r2.pdf',bbox_inches='tight')
  else: plt.show()
  plt.close()

#Plot eigenvalues and eigenproperties
def plot_ed_log(save=True):
  norm = mpl.colors.Normalize(vmin=0, vmax=3.75)
  #FULL EIGENPROPERTIES and EIGENVALUES
  
  beta=2.0
  av_df = pd.read_pickle('analysis/av_sorted_ed_log_d.pickle')
  #av_df = pd.read_pickle('analysis/av_ed_log_d.pickle')

  ## TO GET NICE FORMATTING
  g = sns.FacetGrid(av_df,col='model',col_wrap=3,hue='Sz')

  for model in np.arange(16):
    rgba_color = cm.Blues(norm(3.75-beta))
    rgba_color2 = cm.Oranges(norm(3.75-beta))
    z=-1
    fig, axes = plt.subplots(nrows=2,ncols=6,sharey=True,figsize=(12,6))
    for parm in ['iao_n_3dz2','iao_n_3dpi','iao_n_3dd','iao_n_2pz','iao_n_2ppi','iao_n_4s',
    'iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz','iao_Jsd','iao_Us']:
      z+=1
      ax = axes[z//6,z%6]

      '''
      if(beta==2.0):
        p=parm
        if(parm=='iao_Jsd'): p = 'Jsd'
        if(parm=='iao_Us'):  p = 'Us'
        
        full_df['energy']-=min(full_df['energy'])

        f_df = full_df[full_df['Sz']==0.5]
        x = f_df[p].values
        y = f_df['energy'].values
        yerr = f_df['energy_err'].values
        plt.errorbar(x,y,yerr,fmt='.',c=rgba_color,alpha=0.2)

        f_df = full_df[full_df['Sz']==1.5]
        x = f_df[p].values
        y = f_df['energy'].values
        yerr = f_df['energy_err'].values
        plt.errorbar(x,y,yerr,fmt='.',c=rgba_color2,alpha=0.2)
      '''

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
    if(save): plt.savefig('analysis/sorted_ed_'+str(model)+'_log.pdf',bbox_inches='tight')
    #if(save): plt.savefig('analysis/ed_'+str(model)+'_log.pdf',bbox_inches='tight')
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
  exp_parms_list, yhat, yerr_u, yerr_l = log_fit_bootstrap(X,n=100)
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
def analyze(df,save=False):
  #LOG DATA COLLECTION
  '''
  #Generate all possible models
  X=df[['mo_n_4s','mo_n_2ppi','mo_n_2pz','Jsd','Us']]
  hopping=df[['mo_t_pi','mo_t_dz','mo_t_ds','mo_t_sz']]
  y=df['energy']
  model_list=[list(X)]
  for n in range(1,hopping.shape[1]+1):
    s=set(list(hopping))
    models=list(map(set,itertools.combinations(s,n)))
    model_list+=[list(X)+list(m) for m in models]
  print(len(model_list))

  df0,df1,df2=main_log(df,model_list,betas=[2.0])
  print(df0)
  #print(df1)
  print(df2)
  df0.to_pickle('analysis/regr_log.pickle')
  #df1.to_pickle('analysis/oneparm_log.pickle')
  df2.to_pickle('analysis/ed_log.pickle')
  exit(0)
  '''

  '''
  #Sort/group eigenvalues
  df = pd.read_pickle('analysis/ed_log.pickle')
  df3 = None
  for model in range(16):
    df2 = df[(df['model']==model)]
    for j in range(2,max(df2['bs_index'])+1):
      offset = 0
      for Sz in [0.5,1.5]:
        a = df2[(df2['bs_index']==1)&(df2['Sz']==Sz)]#.iloc[:m[Sz]]
        amat = np.array(list(a['ci']))

        b = df2[(df2['bs_index']==j)&(df2['Sz']==Sz)]#.iloc[:m[Sz]]
        bmat = np.array(list(b['ci']))
    
        #Apply permutation to pair up non degenerate states
        cost = -1.*np.dot(amat,bmat.T)**2
        row_ind, col_ind = linear_sum_assignment(cost)
        
        ovlp = np.dot(amat,bmat.T)
        plt.matshow(ovlp,vmin=-1,vmax=1,cmap=plt.cm.bwr)
        plt.show()

        bmat = bmat[col_ind,:]
    
        ovlp = np.dot(amat,bmat.T)
        plt.matshow(ovlp,vmin=-1,vmax=1,cmap=plt.cm.bwr)
        plt.show()

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
          
          #Check that the degenerate space is actually spanned properly
          if(len_check):
            sub_mat = abdot[sub_ind][:,sub_ind]**2
            sum_1 = np.around(sub_mat.sum(axis=0),2)
            sum_2 = np.around(sub_mat.sum(axis=1),2)
            if(((sum_1 - 1).sum() == 0 ) & ((sum_2 - 1).sum()==0)): sum_check = True
          
          #Check that the eigenvalues are actually degenerate
          if(sum_check):
            degen_a = len(set(np.around(a['energy'].iloc[row_ind[sub_ind]],6)))
            degen_b = len(set(np.around(b['energy'].iloc[col_ind[sub_ind]],6)))
            if((degen_a == 1)&(degen_b ==1)): eig_check = True
          
          #If all checks prevail, then finally assign the elements to be identical
          if(eig_check): bmat[row_ind[sub_ind],:] = amat[row_ind[sub_ind],:]
      
        ovlp = np.dot(amat,bmat.T)
        plt.matshow(ovlp,vmin=-1,vmax=1,cmap=plt.cm.bwr)
        plt.show()
        exit(0)

        #Make sure that we have orthogonal columns and rows
        diff = np.dot(bmat,bmat.T) - np.identity(bmat.shape[0])
        if(abs(np.sum(diff)) > 1e-5):
          print(np.sum(diff),model, Sz)

        #We finally have bmat ordered correctly and everything 
        dtmp = pd.DataFrame({'energy':b['energy'].values[col_ind],'Sz':b['Sz'].values,'bs_index':b['bs_index'].values})
        dtmp['ci']=list(bmat)
        dtmp['eig']=np.arange(dtmp.shape[0]) + offset
        dtmp['model']=b['model'].values
        offset += dtmp.shape[0]
        if(df3 is None): df3 = dtmp
        else: df3 = pd.concat((df3,dtmp),axis=0)
  df3.to_pickle('analysis/sorted_ed_log.pickle')
  exit(0)
  '''

  #Calculate descriptors after sorting
  '''
  from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
  import scipy as sp 

  #df3 = pd.read_pickle('analysis/sorted_ed_log.pickle')
  df3 = pd.read_pickle('analysis/ed_log.pickle')
  df4 = None
  mol = gto.Mole()
  cis = fci.direct_uhf.FCISolver(mol)

  sigUs = []
  sigJsd = []
  sigNdz2 = []
  sigNdpi = []
  sigNdd =[]
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
  df3['iao_n_2pz']=sigN2pz
  df3['iao_n_2ppi']=sigN2ppi
  df3['iao_n_4s']=sigN4s
  df3['iao_t_pi']=sigTpi
  df3['iao_t_ds']=sigTds
  df3['iao_t_dz']=sigTdz
  df3['iao_t_sz']=sigTsz
  df3['iao_Us']=sigUs
  df3['iao_Jsd']=sigJsd

  #df3.to_pickle('analysis/sorted_ed_log_d.pickle')
  df3.to_pickle('analysis/ed_log_d.pickle')
  print(df3)
  exit(0)
  '''

  #Average everything
  ''' 
  df3 = pd.read_pickle('analysis/sorted_ed_log_d.pickle')
  av_df3 = av_ed_log(df3.drop(columns=['ci']))
  av_df3.to_pickle('analysis/av_sorted_ed_log_d.pickle')

  dd = pd.read_pickle('analysis/sorted_ed_log_d.pickle')
  df3 = pd.read_pickle('analysis/ed_log_d.pickle')
  df3['eig'] = dd['eig']
  av_df3 = av_ed_log(df3.drop(columns=['ci']))
  av_df3.to_pickle('analysis/av_ed_log_d.pickle')
  exit(0)
  '''
  
  #Plot ED
  #plot_ed_log()
  #exit(0)

  #Check to see which models have non zero elements
  '''
  dfz = pd.read_pickle('analysis/regr_log.pickle')
  dfz = dfz[dfz['beta']==2.0]
  ind = np.unique(dfz[dfz['Jsd']!=0]['model'])
  print(dfz.iloc[ind])
  print(dfz.iloc[[2,3]])
  exit(0)
  '''

  #Compare spectra between models 
  '''
  df = pd.read_pickle('analysis/av_sorted_ed_log_d.pickle')
  for j in [0,2,3]:
    sub_df = df[df['model']==j]
    sub_df = sub_df.sort_values(by='energy').iloc[:20]
    plt.errorbar(np.zeros(sub_df.shape[0])+j,sub_df['energy'],sub_df['energy_err'],fmt='.')
  plt.xlabel('Model')
  plt.ylabel('Eigenvalues')
  plt.show()
  '''

  #Compare eigenstates between models
  '''
  df = pd.read_pickle('analysis/sorted_ed_log.pickle')
  for j in [0]:
    print(j,'----------')
    for model in [2,3]: #ind:
      print(model,'========')
      for Sz in [0.5,1.5]:
        a = df[(df['model']==j)&(df['Sz']==Sz)]
        amat = np.array(list(a['ci']))[np.arange(30),:]

        b = df[(df['model']==model)&(df['Sz']==Sz)]
        bmat = np.array(list(b['ci']))[np.arange(30),:]
    
        #Apply permutation to pair up non degenerate states
        ovlp = np.dot(amat,bmat.T)**2
        cost = -1*ovlp
        row_ind, col_ind = linear_sum_assignment(cost)
        bmat = bmat[col_ind,:]
        ovlp = np.dot(amat,bmat.T)
        
        plt.matshow(ovlp[:20,:20],vmin=-1,vmax=1,cmap=plt.cm.bwr)
        plt.xticks(np.arange(20),col_ind[:20])
        plt.yticks(np.arange(20),row_ind[:20])
        plt.xlabel('Model '+str(model))
        plt.ylabel('Model 0')
        plt.show()
  exit(0)

  df = pd.read_pickle('analysis/av_sorted_ed_log_d.pickle')
  a = df[(df['model']==0)&(df['Sz']==0.5)]
  b = df[(df['model']==0)&(df['Sz']==1.5)]
  print(a[['energy','iao_n_3dd','iao_n_3dz2','iao_n_3dpi','iao_n_2ppi','iao_n_2pz','iao_n_4s']].iloc[:11])
  print(b[['energy','iao_n_3dd','iao_n_3dz2','iao_n_3dpi','iao_n_2ppi','iao_n_2pz','iao_n_4s']].iloc[:11])
  '''
  
  #X=df[['mo_n_2pz','mo_n_2ppi','mo_n_4s','Jsd','Us','mo_t_pi']]
  #y=df['energy']
  #X=sm.add_constant(X)
  #ols=sm.OLS(y,X).fit()
  #df['resid']=df['energy']-ols.predict()
  #sns.pairplot(df,vars=['energy','iao_t_pi','iao_t_sz'],hue='basestate',markers=['o']+['.']*11)
  #plt.show()

  df['energy']-=(-213.3571109*27.2114)
  df['ind']=[11,12,13]
  print(df[['energy','energy_err','ind']])
  #sns.pairplot(df,vars=['energy','iao_t_pi','iao_t_sz','iao_t_ds','iao_t_dz'],hue='ind')
  #plt.show()

if __name__=='__main__':
  #DATA COLLECTION
  df=collect_df()
  df=format_df(df)
  df.to_pickle('vmc_formatted_gosling.pickle')
  
  #DATA ANALYSIS
  df=pd.read_pickle('vmc_formatted_gosling.pickle')
  analyze(df)
