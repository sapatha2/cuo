import numpy as np 
import statsmodels.api as sm
from scipy.optimize import minimize
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_log_error

import numpy as np

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

def prior_score(b,df):
  df_train = df[df['prior']==False]
  y = df_train['energy'] 
  yhat = pred(b,df_train.drop(columns=['energy','prior']))
  score_train = r2_score(y,yhat)

  df_prior = df[df['prior']==True]
  y = df_prior['energy'] 
  yhat = pred(b,df_prior.drop(columns=['energy','prior']))
  score_prior = (yhat-y).values
  return score_train, score_prior 
  
def pred(b,X):
  return np.dot(b,X.values.T)

def cost(b,X,y,lam,X_prior,y_prior):
  '''
  input:
    beta - coefficients of linear model
    X - training data 
    y - training y 
    lam - lambda for cost 
    X_prior - priors
    y_prior - prior y vals
  return:
    cost - SSE(training)/n_train + lambda* SSE(priors)/n_prior
  '''

  #RMS
  yhat = pred(b,X)
  c1 = np.linalg.norm(pred(b,X) - y)**2

  yhat_prior = pred(b,X_prior)
  c2 = np.sum(sigmoid(y_prior - yhat_prior))
  return c1 + lam*c2

def prior_fit(df,lam):
  '''
  input:
    df - data frame with the following properties 
      1. y - Values, should be labelled "y"
      2. prior - Boolean, should tell us what is a prior and what isn't
      3. X - everything else in the data frame is X
    lam - lambda for the cost function
  return :
    coef - coefficients which minimize cost
  '''

  df_train = df[df['prior']==False]
  df_prior = df[df['prior']==True]
 
  y = df_train['energy']
  X = df_train.drop(columns=['energy','prior'])
  
  y_prior = df_prior['energy']
  X_prior = df_prior.drop(columns=['energy','prior'])

  ols=sm.WLS(y,X).fit()
  b0=ols.params.values

  return minimize(lambda b: cost(b,X,y,lam,X_prior,y_prior), b0).x

if __name__=='__main__':
  #Let's do a test case
  import matplotlib.pyplot as plt

  x = np.linspace(0,10,100)
  y = 2*x + np.random.normal(size=x.shape[0])
  d = pd.DataFrame({'y':y,'x':x})
  d['const'] = 1
  d['prior'] = False

  x_prior = np.arange(-4,1,1)
  y_prior = 5*x_prior
  d_prior = pd.DataFrame({'y':y_prior,'x':x_prior})
  d_prior['const'] = 1
  d_prior['prior'] = True

  df = pd.concat((d,d_prior),axis=0)

  for lam in [0,2,10,1000]:
    params = prior_fit(df,lam)
    plot_x = np.linspace(-10,10,100)
    plot_y = params[0]*plot_x + params[1]
    plt.plot(plot_x,plot_y,label=str(lam))
  plt.plot(x,y,'bo')
  plt.plot(x_prior,y_prior,'ro')
  plt.legend(loc=1)
  plt.show()
