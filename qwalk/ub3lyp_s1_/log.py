import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd 
import statsmodels.api as sm

def pred(b,X):
  return np.dot(b,X.values.T)

def cost(b,X,y,w):
  yhat = pred(b,X)
  mi = min([min(yhat),min(y)])
  c=np.sum(w*np.log((yhat-mi)/(y-mi)))
  return c

def log_fit(df):
  X=df.drop(columns=['energy','weights'])
  y=df['energy']
  w=df['weights']
    
  ols=sm.WLS(y,X,weights=w).fit()
  b0=ols.params.values
  
  res_exp = minimize(lambda b: cost(b,X,y,w), b0).x
  return res_exp, pred(res_exp,X)

def log_fit_bootstrap(df,n=500):
  #Boostrap loop
  yhat=[]
  coef=[]
  for i in range(n):
    dfi=df.sample(n=df.shape[0],replace=True)
    res_expi, __ = log_fit(dfi)
    yhati = pred(res_expi,df.drop(columns=['energy','weights']))
    
    yhat.append(yhati)
    coef.append(res_expi)
  yhat=np.array(yhat)
  coef=np.array(coef)
  
  #Means of predictions and errors
  pred_mu=np.mean(yhat,axis=0)
  #pred_err=np.std(yhat,axis=0)
  
  #Confidence intervals
  pred_err_u = np.percentile(yhat,97.5,axis=0)
  pred_err_l = np.percentile(yhat,2.5,axis=0)

  return coef, pred_mu, pred_err_u, pred_err_l
