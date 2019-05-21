import os 
import numpy as np 
import scipy as sp 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.linear_model
from ed import ED
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce 
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.optimize import linear_sum_assignment 
from find_connect import  *
import matplotlib as mpl 
from prior import prior_fit, prior_score
import itertools 
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
  
  prior_df = avg_eig[(avg_eig['model']==12)&(avg_eig['Sz']==0.5)].iloc[2]
  prior_df = pd.concat((prior_df,avg_eig[(avg_eig['model']==9)&(avg_eig['Sz']==1.5)].iloc[0]),axis=1)
  prior_df = prior_df.T
  prior_df = prior_df[var]
  prior_df['prior'] = True
  prior_df['energy'] = min(df['energy']) + cutoff
  
  return pd.concat((df,prior_df),axis=0)

######################################################################################
#ANALYSIS CODE 
def mahalanobis(df,vec):        
  variables = ['iao_n_4s','iao_n_3dz2','iao_n_3dpi','iao_n_3dd',
  'iao_n_2pz','iao_n_2ppi','iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz']
  dist = df[variables]
  vec = vec[variables]
  return distance.mahalanobis(dist.mean(),vec,np.linalg.inv(dist.cov()))

#ED
def ed(model,exp_parms):
  param_names=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz',
  'mo_t_ds','mo_t_sz','Jsd','Us']
  params=[]
  for parm in param_names:
    if(parm in model): params.append(exp_parms[model.index(parm)])
    else: params.append(0)
  
  norb=9
  nelec=(8,7)
  nroots=30
  res1=ED(params,nroots,norb,nelec)

  nelec=(9,6)
  nroots=30
  res3=ED(params,nroots,norb,nelec)

  E1 = res1[0]
  E3 = res3[0]
  
  return [E1-min(E1),E3-min(E1)]

def sigmoid(x):
  return 1/(1+np.exp(-1*x))

#RUN
def analyze(df=None,save=False):
  #Analysis
  df = add_priors(df,cutoff = 2)
  
  '''
  #Generate models + get all the proeperties we need
  oneparm_df = pd.read_pickle('analysis/oneparm.pickle').drop(columns=['r2_mu','r2_err'])
  prior_df = None
  for i in [5, 9, 12, 21, 20, 24]:
    ind = np.nonzero(oneparm_df.iloc[i])[0]
    model = np.array(list(oneparm_df))[ind]
    model = list(model[:int(len(model)/2)])
    print(model)

    fit_df = df[['energy','prior']+model]
    fit_df['const'] = 1

    lams = np.arange(0,55,5)
    E_err = []
    s1_mu = []
    s1_err = []
    s2_mu = []
    s2_err = []
    r2_mu = []
    r2_err = []
    params_mu = []
    params_err = []
    for lam in lams:
      print("lambda = "+str(lam)) 
      E = []
      r2 = []
      s1 = []
      s2 = []
      ps = []
      for j in range(10): #10 BS samples for error bars
        d = fit_df[fit_df['prior']==False].sample(n=fit_df[fit_df['prior']==False].shape[0],replace=True)
        d = pd.concat((d,fit_df[fit_df['prior']==True]),axis=0)

        params = prior_fit(d,lam)
        score = prior_score(params,d)
        e = ed(model,params[:-1])
  
        E.append(np.array(e[0]+e[1])-min(e[0]+e[1]))
        s1.append(score[1].values[0])
        s2.append(score[1].values[1])
        ps.append(params)
        r2.append(score[0])

      r2_mu.append(np.mean(r2))
      r2_err.append(np.std(r2))
    
      s1_mu.append(np.mean(s1))
      s1_err.append(np.std(s1))

      s2_mu.append(np.mean(s2))
      s2_err.append(np.std(s2))

      E_err.append(np.std(E,axis=0).mean())

      params_mu.append(np.mean(ps,axis=0))
      params_err.append(np.std(ps,axis=0))

    data = pd.DataFrame({'r2_mu':r2_mu,'r2_err':r2_err,'E_err':E_err,'s1_mu':s1_mu,'s1_err':s1_err,
    's2_mu':s2_mu,'s2_err':s2_err,'params_mu':params_mu,'params_err':params_err,'lam':lams,'model':i*np.ones(len(lams))})
    if(prior_df is None): prior_df = data
    else: prior_df = pd.concat((prior_df,data),axis=0)
  prior_df.to_pickle('analysis/prior.pickle')
  exit(0)
  '''

  prior_df = pd.read_pickle('analysis/prior.pickle')

  #R2 vs lambda
  plt.subplot(221)
  for group_model in prior_df.groupby(by='model'):        
    plt.errorbar(group_model[1]['lam'],group_model[1]['r2_mu'],
    group_model[1]['r2_err'],fmt='o-')
  plt.xlabel('lambda')
  plt.ylabel('r2')

  #R2 vs s1
  plt.subplot(222)
  for group_model in prior_df.groupby(by='model'):        
    plt.errorbar(2-group_model[1]['s1_mu'],group_model[1]['r2_mu'],
    yerr=group_model[1]['r2_err'],xerr=group_model[1]['s1_err'],marker='o')
  plt.axvline(2.0,c='k',ls='--')
  plt.xlim(-0.1,2.5)
  plt.xlabel('E_1, eV')
  plt.ylabel('r2')

  #R2 vs s2
  plt.subplot(223)
  for group_model in prior_df.groupby(by='model'):        
    plt.errorbar(group_model[1]['r2_mu'],2-group_model[1]['s2_mu'],
    yerr=group_model[1]['s2_err'],xerr=group_model[1]['r2_err'],marker='o')
  plt.axhline(2.0,c='k',ls='--')
  plt.ylim(-0.1,2.5)
  plt.ylabel('E_2, eV')
  plt.xlabel('r2')

  #s1_err vs s2_err for low energy error models
  plt.subplot(224)
  for group_model in prior_df.groupby(by='model'):        
      plt.errorbar(2-group_model[1]['s1_mu'],2-group_model[1]['s2_mu'],
      yerr=group_model[1]['s2_err'],xerr=group_model[1]['s1_err'],marker='o',label=str(int(group_model[0])))
  plt.legend(loc='best')
  plt.xlabel('E_1, eV')
  plt.ylabel('E_2, eV')
  plt.axvline(2.0,c='k',ls='--')
  plt.xlim(-0.1,2.5)
  plt.axhline(2.0,c='k',ls='--')
  plt.ylim(-0.1,2.5)
  plt.show()

if __name__=='__main__':
  #DATA COLLECTION
  df = pd.read_pickle('formatted_gosling.pickle')
  analyze(df)
