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
pd.options.mode.chained_assignment = None  # default='warn'

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
  exp_parms_list, yhat, yerr_u, yerr_l = log_fit_bootstrap(X,n=100)

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
    exp_parms, yhat, yerr_u, yerr_l = log_fit_bootstrap(X_train,n=100)
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
  for exp_parms in exp_parms_list:
    #Figure out which parameters are in my list
    param_names=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz',
    'mo_t_ds','mo_t_sz','Jsd','Us']
    params=[]
    for parm in param_names:
      if(parm in model): params.append(exp_parms[model.index(parm)+1])
      else: params.append(0)
    
    norb=9
    nelec=(8,7)
    nroots=25
    res1=ED(params,nroots,norb,nelec)

    nelec=(9,6)
    nroots=15
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
    'iao_t_pi':t_pi,'iao_t_ds':t_ds,'iao_t_dz':t_dz,'iao_t_sz':t_sz,'iao_Us':res1[4],'iao_Jsd':res1[5]})

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
    'iao_t_pi':t_pi,'iao_t_ds':t_ds,'iao_t_dz':t_dz,'iao_t_sz':t_sz,'iao_Us':res3[4],'iao_Jsd':res3[5]})),axis=0)

    d['energy']-=min(d['energy'])
    d['index']=np.arange(d.shape[0])

    if(full_df is None): 
      d['eig'] = np.arange(d.shape[0])
      full_df = d
    else: 
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
      full_df = pd.concat((full_df,d),axis=0)
  return full_df

#CIs and means for ED
def av_ed_log(eig_df):
  av_df = None
  eig_df['index']=eig_df.index.values

  for model in range(16):
    for beta in np.arange(0,3.75,0.25):
      for Sz in [0.5,1.5]:
        for eig in range(max(eig_df[eig_df['Sz']==Sz]['index'])):
          sub_df = eig_df[(eig_df['Sz']==Sz)*(eig_df['model']==model)*(eig_df['index']==eig)*(eig_df['beta']==beta)]
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

######################################################################################
#LOG METHODS (PLOTS) 
#Plot regression parameters
def plot_regr_log(save=False):
  full_df=pd.read_pickle('analysis/regr_log.pickle')

  for model in [0,2,3,8,14]:
    print(full_df[(full_df['model']==model)*(full_df['beta']==2.0)].iloc[0])
  exit(0)

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
  full_df=pd.read_pickle('analysis/oneparm_log_noJ.pickle')
  model=[]
  for i in range(16):
    model+=list(np.linspace(i,i+0.75,15))
  full_df['model']=model
 
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

#Plot eigenvalues and eigenproperties
def plot_ed_log(full_df,save=False):
  av_df=pd.read_pickle('analysis/av_ed_log_noJ.pickle')
  g = sns.FacetGrid(av_df,col='model',col_wrap=4,hue='Sz')
  #print(av_df)
  #exit(0)

  norm = mpl.colors.Normalize(vmin=0, vmax=3.75)
  #FULL EIGENPROPERTIES and EIGENVALUES
  for model in np.arange(16): #[0,2,3,8,14]: #np.arange(16):
    for beta in [2.0]:#np.arange(3.75,-0.25,-0.25):
      rgba_color = cm.Blues(norm(3.75-beta))
      rgba_color2 = cm.Oranges(norm(3.75-beta))
      z=0
      for parm in ['iao_n_3d','iao_n_2pz','iao_n_2ppi','iao_n_4s',
      'iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz','iao_Jsd','iao_Us']:
        z+=1
        plt.subplot(2,5,0+z)

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

        sub_df = av_df[(av_df['model']==model)*(av_df['beta']==beta)*(av_df['Sz']==0.5)]
        x=sub_df[parm].values
        xerr=sub_df[parm+'_err'].values
        y=sub_df['energy'].values
        yerr=sub_df['energy_err'].values
        plt.errorbar(x,y,xerr=xerr,yerr=yerr,markeredgecolor='k',fmt='o',c=rgba_color)
       
        sub_df = av_df[(av_df['model']==model)*(av_df['beta']==beta)*(av_df['Sz']==1.5)]
        x=sub_df[parm].values
        xerr=sub_df[parm+'_err'].values
        y=sub_df['energy'].values
        yerr=sub_df['energy_err'].values
        plt.errorbar(x,y,xerr=xerr,yerr=yerr,markeredgecolor='k',fmt='o',c=rgba_color2)
        
        plt.ylim((-0.2,4.5))
        plt.xlabel(parm)
        plt.ylabel('energy (eV)')
    if(save): plt.savefig('analysis/ed_'+str(model)+'_log_noJ.pdf',bbox_inches='tight')
    else: plt.show()
    plt.clf()
  '''
  av_df = pd.read_pickle('analysis/ed_log.pickle')
  beta=1.0
  for model in [0,4]: 
    z=0
    for parm in ['iao_n_3d','iao_n_2pz','iao_n_2ppi','iao_n_4s',
    'iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz','iao_Jsd','iao_Us']:
      z+=1
      plt.subplot(2,5,0+z)
      sub_df = av_df[(av_df['model']==model)*(av_df['beta']==beta)]
      sns.scatterplot(x=parm,y='energy',data=sub_df,hue='Sz')
    plt.show()
  '''
  '''
      sub_df = av_df[(av_df['model']==model)*(av_df['beta']==beta)*(av_df['Sz']==1.5)]
      x=sub_df[parm].values
      y=sub_df['energy'].values
      plt.errorbar(x,y,markeredgecolor='k',fmt='.',c=rgba_color2)
      
      plt.ylim((-0.2,4.5))
      plt.xlabel(parm)
      plt.ylabel('energy (eV)')
  if(save): plt.savefig('analysis/ed_'+str(model)+'_full_log.pdf',bbox_inches='tight')
  else: plt.show()
  '''

  '''

  #COMBINED
  norm = mpl.colors.Normalize(vmin=0, vmax=3.75)
  #FULL EIGENPROPERTIES and EIGENVALUES
  markers=['o','','s','<','','','','','^','','','','','','>']
  for model in np.arange(16): #[0,2,3,8,14]: #np.arange(16):
    for beta in [2.0]:#np.arange(3.75,-0.25,-0.25):
      rgba_color = cm.Blues(norm(3.75-beta))
      rgba_color2 = cm.Oranges(norm(3.75-beta))
      z=0
      for parm in ['iao_n_3d','iao_n_2pz','iao_n_2ppi','iao_n_4s',
      'iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz','iao_Jsd','iao_Us']:
        z+=1
        plt.subplot(2,5,0+z)

        sub_df = av_df[(av_df['model']==model)*(av_df['beta']==beta)*(av_df['Sz']==0.5)].iloc[:25]
        x=sub_df[parm].values
        xerr=sub_df[parm+'_err'].values
        y=sub_df['energy'].values
        yerr=sub_df['energy_err'].values
        plt.errorbar(x,y,xerr=xerr,yerr=yerr,markeredgecolor='k',fmt=markers[model],c=rgba_color)
       
        sub_df = av_df[(av_df['model']==model)*(av_df['beta']==beta)*(av_df['Sz']==1.5)].iloc[:15]
        x=sub_df[parm].values
        xerr=sub_df[parm+'_err'].values
        y=sub_df['energy'].values
        yerr=sub_df['energy_err'].values
        plt.errorbar(x,y,xerr=xerr,yerr=yerr,markeredgecolor='k',fmt=markers[model],c=rgba_color2)
       
        plt.ylim((-0.2,4.5))
        plt.xlabel(parm)
        plt.ylabel('energy (eV)')
  plt.savefig('analysis/ed_combined_log.pdf',bbox_inches='tight')
  plt.clf()
  '''
  return 0

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

def plot_noiser2(save=False):
  ed_df=pd.read_pickle('analysis/av_ed_log_noJ.pickle')
  r2_df=pd.read_pickle('analysis/oneparm_log_noJ.pickle')

  betas=[]
  models=[]
  e_errs=[]
  p_errs=[]
  r2s=[]
  r2_errs=[]
  rmsebars=[]
  rmsebar_errs=[]
  for beta in np.arange(0,3.75,0.25):
    for model in range(16):
      a=ed_df[(ed_df['beta']==beta)*(ed_df['model']==model)]
      b=r2_df[(r2_df['beta']==beta)*(r2_df['model']==model)]
      betas.append(beta)
      models.append(model)
      r2s.append(b['R2cv_mu_test'])
      r2_errs.append(b['R2cv_std_test'])
      rmsebars.append(b['RMSEbarcv_mu_test'])
      rmsebar_errs.append(b['RMSEbarcv_std_test'])
      e_errs.append(np.sum(a['energy_err'].values))
      p_err=0
      for col in ['iao_n_3d', 'iao_n_2pz', 'iao_n_2ppi', 'iao_n_4s',
      'iao_t_pi', 'iao_t_ds', 'iao_t_dz', 'iao_t_sz', 'iao_Us', 'iao_Jsd']:
        p_err += np.sum(a[col+'_err'].values)
      p_errs.append(p_err)

  ret_df = pd.DataFrame(columns=['beta','model','R2cv_mu_test','R2cv_std_test','RMSEbarcv_mu_test','RMSEbarcv_std_test',
  'e_err','p_err'],
  data=np.array([betas,models,r2s,r2_errs,rmsebars,rmsebar_errs,e_errs,p_errs]).T)

  markers=['.','o','v','^','<','>','8','s','P','p','h','H','*','X','D','d']
  for error in ['e_err','p_err']:
    gs = mpl.gridspec.GridSpec(1, 2,width_ratios=[15,1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    norm = mpl.colors.Normalize(vmin=0, vmax=3.75)
    
    for beta in [2.0]:
      #cmap = cm.nipy_spectral
      for model in range(16):
        pdf=ret_df[(ret_df['model']==model)*(ret_df['beta']==beta)]
        ax1.errorbar(pdf[error].values,pdf['R2cv_mu_test'].values,pdf['R2cv_std_test'].values,c='k',marker=markers[model],
        label='model '+str(model))#cmap(norm(beta)),marker=markers[model],label='model '+str(model))
        
        #ax1.errorbar(pdf[error].values,pdf['RMSbarcv_mu_test'].values,pdf['RMSEbarcv_std_test'].values,c='k',marker=markers[model],
        #label='model '+str(model))#cmap(norm(beta)),marker=markers[model],label='model '+str(model))
    '''
    for model in np.arange(16):
      pdf=ret_df[(ret_df['model']==model)]
      ax1.plot(pdf[error].values,pdf['RMSEbarcv_mu_test'].values,c='gray',ls='--')
    '''

    #cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=cmap, norm=norm,orientation='vertical')
    ax1.legend(loc='best')
    ax1.set_xlabel(error)
    ax1.set_ylabel('5-fold CV RMSE bar')
    if(save): plt.savefig('analysis/rmsebarvs'+error+'.pdf',bbox_inches='tight')
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

######################################################################################
#RUN
def analyze(df,save=False):
  #LOG DATA COLLECTION
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

  df0,df1,df2=main_log(df,model_list)
  #df3=av_ed_log(df2)
  print(df0)
  print(df1)
  print(df2)
  df0.to_pickle('analysis/regr_log.pickle')
  df1.to_pickle('analysis/oneparm_log.pickle')
  df2.to_pickle('analysis/ed_log.pickle')
  #df3.to_pickle('analysis/av_ed_log.pickle')
  exit(0)

  #df = pd.read_pickle('analysis/ed_log.pickle')
  #df=pd.read_pickle('analysis/av_ed_log.pickle')
  #print(df)
  #exit(0)

  #av_ed_log(pd.read_pickle('analysis/ed_log.pickle')).to_pickle('analysis/av_ed_log.pickle')
  #plot_noiser2(save=False)

  #LOG PLOTTING
  #plot_oneparm_valid_log(save=False)
  #plot_Xerr(df)
  #plot_regr_log(save=False)
  #plot_ed_log(df,save=True) 
  '''
  beta=2.0
  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','Jsd','Us']
  weights=np.exp(-beta*(df['energy']-min(df['energy'])))
  X=df[model+['energy','energy_err']+['Sz','basestate']]
  X=sm.add_constant(X)
  X['weights']=weights
  plot_fit_log(X,save=True,fname='analysis/regr_log')
  '''
if __name__=='__main__':
  #DATA COLLECTION
  #df=collect_df()
  #df=format_df(df)
  #df.to_pickle('formatted_gosling.pickle')
  #exit(0)

  #DATA ANALYSIS
  df=pd.read_pickle('formatted_gosling.pickle')
  analyze(df)
