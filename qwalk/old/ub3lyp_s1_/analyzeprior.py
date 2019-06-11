import os 
import numpy as np 
import scipy as sp 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.linear_model
from ed import ED, h1_moToIAO
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce 
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.optimize import linear_sum_assignment 
from find_connect import  *
import matplotlib as mpl 
from prior import prior_fit, prior_score #, sigmoid
import itertools 
from analyzedmc import sort_ed, desc_ed, avg_ed, plot_ed_small, comb_plot_ed_small, plot_ed
######################################################################################
#FROZEN METHODS
#Collect df
#ADD PRIORS
def add_priors(df,cutoff):
  var = ['energy','iao_n_3dd','iao_n_3dpi','iao_n_3dz2','iao_n_2pz','iao_n_2ppi','iao_n_4s',
  'iao_t_pi','iao_t_sz','iao_t_dz','iao_t_ds','Jsd','Us']+\
  ['mo_n_3dd','mo_n_3dpi','mo_n_3dz2','mo_n_2pz','mo_n_2ppi','mo_n_4s',
  'mo_t_pi','mo_t_sz','mo_t_dz','mo_t_ds']
  avg_eig = pd.read_pickle('analysis/avg_eig.pickle')
  
  df = df[var]
  df['prior'] = False
  
  prior_df = pd.read_pickle('analysis/outlier.pickle')
  prior_df = prior_df.iloc[[-5,-3,-1,5]] #Remove any degenerate/near degenerate stuff
  prior_df['prior'] = True 
  prior_df['energy'] = min(df['energy']) + cutoff

  return pd.concat((df,prior_df),axis=0)

def get_iao_parms(exp_parms,model):
  param_names=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz',
  'mo_t_sz','mo_t_ds','Jsd','Us']
  params=[]
  for parm in param_names:
    if(parm in model): params.append(exp_parms[model.index(parm)])
    else: params.append(0)
  
  #params input must be es,epi,epz,tpi,tdz,tsz,tds
  h1_iao = h1_moToIAO(params[:-2])
  #IAO ordering: ['del','del','yz','xz','x','y','z2','z','s']
  #Return parameter ordering: iao_n_3dz2, ,iao_n_3dpi, iao_n_3dd, 
  #iao_n_4s, iao_n_2ppi, iao_n_2pz, iao_t_pi, 
  #iao_t_dz, iao_t_sz, iao_t_ds, Jsd, Us

  return [h1_iao[6,6], h1_iao[2,2], h1_iao[0,0],
  h1_iao[8,8],h1_iao[4,4],h1_iao[7,7],h1_iao[2,5],
  h1_iao[6,7],h1_iao[7,8],h1_iao[6,8],params[-2],params[-1]]

######################################################################################
#ANALYSIS CODE 
def prior_analysis(df,cutoff=2):
  #Generate models + get all the properties we need
  df = add_priors(df,cutoff = 2)
  oneparm_df = pd.read_pickle('analysis/oneparm.pickle').drop(columns=['r2_mu','r2_err'])
  prior_df = None
  for i in [5, 9, 12]:#, 21, 20, 24]:
    ind = np.nonzero(oneparm_df.iloc[i])[0]
    model = np.array(list(oneparm_df))[ind]
    model = list(model[:int(len(model)/2)])
    print(model)

    fit_df = df[['energy','prior']+model]
    fit_df['const'] = 1

    lams = np.arange(0,22,2)
    s_mu = []
    s_err = []
    r2_mu = []
    r2_err = []
    params_mu = []
    params_err = []
    for lam in lams:
      print("lambda = "+str(lam)) 
      E = []
      r2 = []
      s = []
      ps = []
      for j in range(10): #10 BS samples for error bars
        d = fit_df[fit_df['prior']==False].sample(n=fit_df[fit_df['prior']==False].shape[0],replace=True)
        d = pd.concat((d,fit_df[fit_df['prior']==True]),axis=0)

        params = prior_fit(d,lam)
        score = prior_score(params,d)
        print(score[0],score[1].values)

        s.append(score[1])
        ps.append(params)
        r2.append(score[0])

      r2_mu.append(np.mean(r2))
      r2_err.append([np.percentile(r2,2.5),np.percentile(r2,97.5)])
   
      s_mu.append(np.mean(s,axis=0))
      s_err.append([np.percentile(s,2.5,axis=0),np.percentile(s,97.5,axis=0)])
      print(len(s_mu[-1]))
      print(len(s_err[-1][0]))

      params_mu.append(np.mean(ps,axis=0))
      params_err.append(np.std(ps,axis=0))

    data = pd.DataFrame({'r2_mu':r2_mu,'r2_err':r2_err,'s_mu':s_mu,'s_err':s_err,
    'params_mu':params_mu,'params_err':params_err,'lam':lams,'model':i*np.ones(len(lams))})
    if(prior_df is None): prior_df = data
    else: prior_df = pd.concat((prior_df,data),axis=0)
  return prior_df 

def plot_prior():
  #Plot prior analysis figure
  prior_df = pd.read_pickle('analysis/prior.pickle')

  #Calculate model scores
  scores = []
  scores_u = []
  scores_d = []
  for i in range(prior_df.shape[0]):
    y = 2 - prior_df['s_mu'].iloc[i]
    y = y[y>0]
    y = np.linalg.norm(y)**2
    y = y.sum()
    scores.append(y)
    
    y = 2 - prior_df['s_err'].iloc[i][0]
    y = y[y>0]
    y = np.linalg.norm(y)**2
    y = y.sum()
    scores_u.append(y)

    y = 2 - prior_df['s_err'].iloc[i][1]
    y = y[y>0]
    y = np.linalg.norm(y)**2
    y = y.sum()
    scores_d.append(y)
  
  prior_df['score'] = scores
  prior_df['score_u'] = scores_u
  prior_df['score_d'] = scores_d

  #R2 vs lambda
  fig = plt.figure(figsize = (3,6))
  plt.subplot(211)
  for group_model in prior_df.groupby(by='model'):        
    error_r2 = np.vstack(group_model[1]['r2_err']).T
    error_r2[0] = group_model[1]['r2_mu'] - error_r2[0,:]
    error_r2[1] = -group_model[1]['r2_mu'] + error_r2[1,:]
    plt.errorbar(group_model[1]['lam'],group_model[1]['r2_mu'],
    yerr=error_r2,fmt='o-')
  plt.xticks(np.arange(0,24,4),['']*len(np.arange(0,24,4)))
  plt.ylabel(r'R$^2$')

  #Lambda vs score
  labels={5:r'Min',9:r'Min$ + \bar{c}_{d_\pi}^\dagger \bar{c}_{p_\pi}$',
  12:r'Min$ + \bar{c}_{d_{z^2}}^\dagger \bar{c}_{p_z}$',
  20:r'Min$ + \bar{c}_{d_\pi}^\dagger \bar{c}_{p_\pi}, \bar{c}_{4s}^\dagger \bar{c}_{p_z}$',
  21:r'Min$ + \bar{c}_{d_\pi}^\dagger \bar{c}_{p_\pi}, \bar{c}_{d_z^2}^\dagger \bar{c}_{4s}$',
  24:r'Min$ + \bar{c}_{d_z^2}^\dagger \bar{c}_{p_z}, \bar{c}_{d_z^2}^\dagger \bar{c}_{4s}$'}
  plt.subplot(212)
  for group_model in prior_df.groupby(by='model'):        
    score_u = group_model[1]['score_u'] - group_model[1]['score']
    score_d = group_model[1]['score'] - group_model[1]['score_d']
    plt.errorbar(group_model[1]['lam'],group_model[1]['score'],
    yerr=[score_d,score_u],marker='o',ls='None',label=labels[group_model[0]])
  plt.legend(loc='best')
  plt.xticks(np.arange(0,24,4))
  plt.ylabel(r'QHL (eV$^2$)')
  plt.xlabel(r'$\lambda$')
  plt.savefig('analysis/figs/prior.pdf',bbox_inches='tight')
  return -1

def exact_diag_prior(df, cutoff, model_ind, lam, nbs=20):
  #Do ED for our model
  max_model = ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz','mo_t_sz','mo_t_ds','Jsd','Us']
  oneparm_df = pd.read_pickle('analysis/oneparm.pickle').drop(columns=['r2_mu','r2_err'])
  df = add_priors(df,cutoff = 2)
  eig_df = None

  model = oneparm_df.iloc[model_ind]
  model = np.nonzero(model)[0]
  model = model[:int(len(model)/2)]
  model_strings = np.array(list(oneparm_df))[model]
  print(model,model_strings)

  fit_df = df[['energy','prior']+list(model_strings)]
  fit_df['const'] = 1
  for n in range(nbs):
    print(n)
    d = fit_df[fit_df['prior']==False].sample(n=fit_df[fit_df['prior']==False].shape[0],replace=True)
    d = pd.concat((d,fit_df[fit_df['prior']==True]),axis=0)

    ps = prior_fit(d,lam)
    params = np.zeros(len(max_model))
    params[model] = ps[:-1]
    
    #Do ED
    norb=9
    nelec=(8,7)
    nroots=30
    res1=ED(params,nroots,norb,nelec)

    nelec=(9,6)
    nroots=30
    res3=ED(params,nroots,norb,nelec)
    
    E = res1[0]
    Sz = np.ones(len(E))*0.5
    ci=np.array(res1[1])
    ci=np.reshape(ci,(ci.shape[0],ci.shape[1]*ci.shape[2]))
    d = pd.DataFrame({'energy':E,'Sz':Sz})
    d['ci']=list(ci)

    E = res3[0]
    Sz = np.ones(len(E))*1.5
    ci=np.array(res3[1])
    ci=np.reshape(ci,(ci.shape[0],ci.shape[1]*ci.shape[2]))
    d2 = pd.DataFrame({'energy':E,'Sz':Sz})
    d2['ci']=list(ci)

    d=pd.concat((d,d2),axis=0)
    d['model'] = model_ind
    d['bs_index'] = n 
    d['energy'] += ps[-1]

    #Concatenate
    if(eig_df is None): eig_df = d 
    else: eig_df = pd.concat((eig_df,d),axis=0)
  return eig_df

def regr_prior(df,model_ind,lam,cutoff=2,nbs=20):
  #Plot regr for our model
  oneparm_df = pd.read_pickle('analysis/oneparm.pickle').drop(columns=['r2_mu','r2_err'])
  df = add_priors(df,cutoff = 2)
  eig_df = None

  model = oneparm_df.iloc[model_ind]
  model = np.nonzero(model)[0]
  model = model[:int(len(model)/2)]
  model_strings = np.array(list(oneparm_df))[model]
  print(model,model_strings)

  fit_df = df[['energy','prior']+list(model_strings)]
  fit_df['const'] = 1
  fit_df['index'] = np.arange(fit_df.shape[0])
  
  y = []
  ind = []
  for n in range(20):
    d = fit_df[fit_df['prior']==False].sample(n=fit_df[fit_df['prior']==False].shape[0],replace=True)
    d = pd.concat((d,fit_df[fit_df['prior']==True]),axis=0)

    ps = prior_fit(d.drop(columns=['index']),lam)
    yhat = np.dot(d.drop(columns=['index','prior','energy']).iloc[:-6],ps)
    y+=list(yhat)
    ind+=list(d.iloc[:-6]['index'].values)

  df = pd.read_pickle('formatted_gosling.pickle')
  plot_df = pd.DataFrame({'y':y,'ind':ind})
  av_df = None
  for sub in plot_df.groupby(by='ind'):
    y_mu = np.mean(sub[1]['y'])
    y_l = y_mu - np.percentile(sub[1]['y'],2.5)
    y_u = np.percentile(sub[1]['y'],97.5) - y_mu
    z = pd.DataFrame({'y_mu':y_mu,'y_l':y_l,'y_u':y_u,'ind':sub[0],
      'y':df.iloc[sub[0]]['energy'],'y_err':df.iloc[sub[0]]['energy_err'],
      'basestate':df.iloc[sub[0]]['basestate'],'Sz':df.iloc[sub[0]]['Sz']},index=[0])
    if(av_df is None): av_df = z
    else: av_df = pd.concat((av_df,z),axis=0)
 
  fig = plt.figure(figsize=(3,3))
  plt.errorbar(av_df['y_mu'].values,av_df['y'].values,yerr=av_df['y_err'].values,
    xerr=[av_df['y_l'].values,av_df['y_u'].values],fmt='.',label=r'DMC')
  plt.plot([-5809.5,-5803.5],[-5809.5,-5803.5],c='k',ls='--')
  plt.legend(loc='best')
  plt.ylabel(r'Min model, $\lambda$=20')
  plt.xlabel(r'E$_{eff}$, eV')
  plt.ylabel(r'E$_{DMC}$, eV')
  plt.ylim((-5809.5,-5803.5))
  plt.xlim((-5809.5,-5803.5))
  plt.xticks([-5809,-5807,-5805])
  plt.yticks([-5809,-5807,-5805])
  plt.savefig('analysis/figs/regr_m'+str(model_ind)+'_l'+str(lam)+'.pdf',bbox_inches='tight')
  return -1

def iao_analysis(df,cutoff=2):
  #Generate models + get iao properties we need
  df = add_priors(df,cutoff = 2)
  oneparm_df = pd.read_pickle('analysis/oneparm.pickle').drop(columns=['r2_mu','r2_err'])
  prior_df = None
  for i in [5, 9, 12, 21, 20, 24]:
    ind = np.nonzero(oneparm_df.iloc[i])[0]
    model = np.array(list(oneparm_df))[ind]
    model = list(model[:int(len(model)/2)])
    print(model)

    fit_df = df[['energy','prior']+model]
    fit_df['const'] = 1

    lams = [20]
    params_mu = []
    params_err = []
    params_iao_mu = []
    params_iao_err = []
    for lam in lams:
      print("lambda = "+str(lam)) 
      ps = []
      ps_iao = []
      for j in range(10): #10 BS samples for error bars
        d = fit_df[fit_df['prior']==False].sample(n=fit_df[fit_df['prior']==False].shape[0],replace=True)
        d = pd.concat((d,fit_df[fit_df['prior']==True]),axis=0)

        params = prior_fit(d,lam)
        params_iao = get_iao_parms(params[:-1],model)
        score = prior_score(params,d)
        print(params_iao)

        ps.append(params)
        ps_iao.append(params_iao)

      params_mu.append(np.mean(ps,axis=0))
      params_err.append(np.std(ps,axis=0))

      params_iao_mu.append(np.mean(ps_iao,axis=0))
      params_iao_err.append(np.std(ps_iao,axis=0))

    data = pd.DataFrame({'params_mu':params_mu,'params_err':params_err,
    'params_iao_mu':params_iao_mu,'params_err':params_iao_err,
    'lam':lams,'model':i*np.ones(len(lams))})
    if(prior_df is None): prior_df = data
    else: prior_df = pd.concat((prior_df,data),axis=0)
  return prior_df 

def analyze(df=None,save=False):
  cutoff = 2

  #Generate models and plot cost 
  '''
  prior_df = prior_analysis(df,cutoff=cutoff)
  prior_df.to_pickle('analysis/prior.pickle')
  plot_prior()
  exit(0)
  '''

  #ED for models
  '''
  for model in [5,9,12]:
    for lam in [20]: #[4,10,20]:
      ed_df = exact_diag_prior(df, cutoff, model, lam, nbs=20)
      ed_df = sort_ed(ed_df)
      ed_df = desc_ed(ed_df).drop(columns=['ci'])
      avg_df = avg_ed(ed_df)
    avg_df.to_pickle('analysis/avg_eig_prior_m'+str(model)+'_l'+str(lam)+'.pickle')
  exit(0)
  '''

  #Combined plot of ED
  avg_eig_df = None
  for model in [5,9,12]:
    lam = 20
    a = pd.read_pickle('analysis/avg_eig_prior_m'+str(model)+'_l'+str(lam)+'.pickle')
    if(avg_eig_df is None): avg_eig_df = a
    else: avg_eig_df = pd.concat((avg_eig_df,a),axis=0)
  comb_plot_ed_small(df,avg_eig_df,[5,9,12],fname='analysis/figs/final_ed.pdf',cutoff=8)
  exit(0)

  #Linear regression plot of selected model 
  model = 5
  lam = 20
  regr_prior(df,model,lam)

  #Model parameters
  params_df = iao_analysis(df)
  params_df.to_pickle('analysis/params.pickle')

if __name__=='__main__':
  #DATA COLLECTION
  df = pd.read_pickle('formatted_gosling.pickle')
  analyze(df)
