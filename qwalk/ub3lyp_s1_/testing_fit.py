import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def pred(b,x):
  return b[0] + b[1]*x

def r(u,tau):
  return np.abs(tau - (u<0).astype(int))*u**2

def cost(b,x,y,s):
  yhat = pred(b,x)
  res = y - yhat
  z=(y-min(y))/s

  #tau=1/2 for z<1, energy under the scale we set
  c=np.sum(r(res[z<1],0.5))
  #tau=1 for z>1, energy above the scale we set 
  c+=np.sum(r(res[z>=1],1.0))
  return c

x=np.arange(100)
y=2*x + np.random.normal(size=x.shape[0])
y[20:]+=np.random.normal(size=80)*y[20:]/10
b0=[0,2]
s=40

res_exp = minimize(lambda b: cost(b,x,y,s), b0).x
res_ols = minimize(lambda b: cost(b,x,y,1e200), b0).x
plt.plot(x,y,'ob',label='DATA')
plt.plot(x,res_exp[0]+res_exp[1]*x,'-g',label='EXP')
plt.plot(x,res_ols[0]+res_ols[1]*x,'-k',label='OLS')
plt.legend(loc='best')
plt.savefig('testing_fit.pdf')
