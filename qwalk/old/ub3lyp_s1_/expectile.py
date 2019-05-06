import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd 
import statsmodels.api as sm

def pred(b,X):
  return np.dot(b,X.values.T)

def r(u,tau):
  return np.abs(tau - (u<0).astype(int))*u**2

def cost(b,X,y,s,w):
  yhat = pred(b,X)
  res = y - yhat
  z=(y-min(y))/s

  #tau=1/2 for z<1, energy under the scale we set
  c=np.sum(w[z<1]*r(res[z<1],0.5))
  #tau=1 for z>1, energy above the scale we set 
  c+=np.sum(w[z>=1]*r(res[z>=1],1.0))
  return c

def expectile_fit(df,s,n):
  #Boostrap loop
  yhat=[]
  coef=[]
  for i in range(n):
    dfi=df.sample(n=df.shape[0],replace=True)
    Xi=dfi.drop(columns=['energy','weights'])
    yi=dfi['energy']
    wi=dfi['weights']

    olsi=sm.WLS(yi,Xi,weights=wi).fit()
    b0i=olsi.params.values
    print("OLS: ",i,b0i)
    
    res_expi = minimize(lambda b: cost(b,Xi,yi,s,wi), b0i).x
    print("EXP: ",i,res_expi)
    
    yhati = pred(res_expi,df.drop(columns=['energy','weights']))
    
    yhat.append(yhati)
    coef.append(res_expi)
  yhat=np.array(yhat)
  coef=np.array(coef)
  
  #Means of predictions and errors
  pred_mu=np.mean(yhat,axis=0)
  pred_err=np.std(yhat,axis=0)
  
  return coef, pred_mu, pred_err
