import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd 
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

def pred(b,x):
  return b[0] + b[1]*x

def r(u,tau):
  return np.abs(tau - (u<0).astype(int))*u**2

def cost(b,x,y,s,w=None):
  if(w is None): w=np.ones(len(y))
  yhat = pred(b,x)
  res = y - yhat
  z=(y-min(y))/s

  #tau=1/2 for z<1, energy under the scale we set
  c=np.sum(w[z<1]*r(res[z<1],0.5))
  #tau=1 for z>1, energy above the scale we set 
  c+=np.sum(w[z>=1]*r(res[z>=1],1.0))
  return c

x=np.arange(100)
y=2*x + np.random.normal(size=x.shape[0])
y[20:]+=np.random.normal(size=80)*y[20:]/10
b0=[0,2]
s=40
w=np.exp(-(1./s)*(y-min(y)))

res_exp = minimize(lambda b: cost(b,x,y,s,w), b0).x
#res_ols = minimize(lambda b: cost(b,x,y,1e200,w), b0).x

X=pd.DataFrame({'X':x})
X=sm.add_constant(X)
ols=sm.WLS(y,X,weights=w).fit()
__, l, u = wls_prediction_std(ols,alpha=0.10)
pred_err=(u-l)/2

plt.plot(x,y,'ob',label='DATA')
plt.errorbar(x,res_exp[0]+res_exp[1]*x,xerr=pred_err,fmt='-g',label='EXP')
plt.errorbar(x,ols.predict(),xerr=pred_err,fmt='-k',label='OLS')
plt.legend(loc='best')
plt.show()
#plt.savefig('testing_fit.pdf')
