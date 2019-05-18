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

#RUN
def analyze(df=None,save=False):
  #Analysis
  df = add_priors(df,cutoff = 2)
  
  oneparm_df = pd.read_pickle('analysis/oneparm.pickle').drop(columns=['r2_mu','r2_err'])
  for i in [5, 9, 12]:
    ind = np.nonzero(oneparm_df.iloc[i])[0]
    model = np.array(list(oneparm_df))[ind]
    model = list(model[:int(len(model)/2)])
    print(model)

    fit_df = df[['energy','prior']+model]
    fit_df['const'] = 1

    lams = np.arange(0,140,20)
    r2scores = []
    diff = []
    for lam in lams:
      params = prior_fit(fit_df,lam)
      print(np.around(params[:-1],2))
      scores = prior_score(params,fit_df)
      r2scores.append(scores[0])
      diff.append(scores[1].sum())
    plt.plot(diff,r2scores,'o-',label=str(i))
  plt.ylabel('R2 score')
  plt.xlabel('E_p - E_cut, eV')
  plt.legend(loc='best')
  plt.show()

if __name__=='__main__':
  #DATA COLLECTION
  df = pd.read_pickle('formatted_gosling.pickle')
  analyze(df)
