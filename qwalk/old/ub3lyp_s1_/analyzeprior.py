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
from prior import prior_fit, prior_score #, sigmoid
import itertools 
from analyzedmc import sort_ed, desc_ed, avg_ed, plot_ed_small
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
  
  '''
  prior_df = avg_eig[(avg_eig['model']==12)&(avg_eig['Sz']==0.5)].iloc[2]
  prior_df = pd.concat((prior_df,avg_eig[(avg_eig['model']==9)&(avg_eig['Sz']==1.5)].iloc[0]),axis=1)
  prior_df = prior_df.T
  prior_df = prior_df[var]
  prior_df['prior'] = True
  '''
  
  prior_df = pd.read_pickle('analysis/outlier.pickle')
  prior_df = prior_df.iloc[[0,11,19,20,22]] #Remove any degenerate/near degenerate stuff
  prior_df['prior'] = True 
  prior_df['energy'] = min(df['energy']) + cutoff

  return pd.concat((df,prior_df),axis=0)

######################################################################################
#ANALYSIS CODE 
def prior_analysis(df,cutoff=2):
  #Generate models + get all the properties we need
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

    #lams = np.arange(0,55,5)
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
  from sklearn.metrics import log_loss
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

  '''
  #R2 vs Score
  plt.subplot(221)
  for group_model in prior_df.groupby(by='model'):        
    error_r2 = np.vstack(group_model[1]['r2_err']).T
    error_r2[0] = group_model[1]['r2_mu'] - error_r2[0,:]
    error_r2[1] = -group_model[1]['r2_mu'] + error_r2[1,:]
    plt.errorbar(group_model[1]['score'],group_model[1]['r2_mu'],
    yerr = error_r2,fmt='o-',label=str(group_model[0]),ls='None')
  plt.xlabel('Quad hinge loss')
  plt.ylabel('r2')
  plt.legend(loc='best')
  '''

  #R2 vs lambda
  plt.subplot(211)
  for group_model in prior_df.groupby(by='model'):        
    error_r2 = np.vstack(group_model[1]['r2_err']).T
    error_r2[0] = group_model[1]['r2_mu'] - error_r2[0,:]
    error_r2[1] = -group_model[1]['r2_mu'] + error_r2[1,:]
    plt.errorbar(group_model[1]['lam'],group_model[1]['r2_mu'],
    yerr=error_r2,fmt='o-',label=str(group_model[0]))
  plt.xticks(np.arange(0,22,2))
  plt.ylabel('r2')

  #Lambda vs score
  plt.subplot(212)
  for group_model in prior_df.groupby(by='model'):        
    score_u = group_model[1]['score_u'] - group_model[1]['score']
    score_d = group_model[1]['score'] - group_model[1]['score_d']
    plt.errorbar(group_model[1]['lam'],group_model[1]['score'],
    yerr=[score_d,score_u],marker='o',ls='None',label=str(group_model[0]))
  plt.legend(loc='best')
  plt.xticks(np.arange(0,22,2))
  plt.ylabel('Quad hinge loss')
  plt.xlabel('lam')

  plt.show()
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
      'basestate':df.iloc[sub[0]]['basestate']},index=[0])
    if(av_df is None): av_df = z
    else: av_df = pd.concat((av_df,z),axis=0)
  
  plt.errorbar(av_df['y_mu'].values,av_df['y'].values,yerr=av_df['y_err'].values,
    xerr=[av_df['y_l'].values,av_df['y_u'].values],fmt='s',alpha=0.5)
  plt.errorbar(av_df['y_mu'].values,av_df['y'].values,yerr=av_df['y_err'].values,
    xerr=[av_df['y_l'].values,av_df['y_u'].values],fmt='s',alpha=0.0)
  ind = av_df['basestate']<0
  plt.errorbar(av_df[ind]['y_mu'].values,av_df[ind]['y'].values,yerr=av_df[ind]['y_err'].values,
    xerr=[av_df[ind]['y_l'].values,av_df[ind]['y_u'].values],fmt='o',markeredgecolor='k',label='Basestate')
  plt.plot(av_df['y'].values,av_df['y'].values,c='k',ls='--')
  plt.legend(loc='best')
  plt.xlabel('E_m, eV')
  plt.ylabel('E_DMC, eV')
  plt.show()
  return -1

def analyze(df=None,save=False):
  cutoff = 2
  #Generate models and plot cost 
  #prior_df = prior_analysis(df,cutoff=cutoff)
  #prior_df.to_pickle('analysis/prior.pickle')
  #plot_prior()
  #exit(0)

  #ED plot of selected model
  model = 5
  lam = 6
  
  for model in [9,20,21,12,24,5]:
    for lam in [4,10,20]:
      ed_df = exact_diag_prior(df, cutoff, model, lam, nbs=20)
      ed_df = sort_ed(ed_df)
      ed_df = desc_ed(ed_df).drop(columns=['ci'])
      avg_df = avg_ed(ed_df)
      avg_df.to_pickle('analysis/avg_eig_prior_m'+str(model)+'_l'+str(lam)+'.pickle')
  
  #plot_ed_small(df,avg_df,model)
  #Linear regression plot of selected model 
  #regr_prior(df,model,lam)

if __name__=='__main__':
  #DATA COLLECTION
  df = pd.read_pickle('formatted_gosling.pickle')
  analyze(df)
