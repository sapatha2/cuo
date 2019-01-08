import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
import statsmodels.api as sm 
#from sklearn.linear_model import OrthogonalMatchingPursuit
#from sklearn.model_selection import cross_val_score

#Load
detgen='s'
n='100'
ndet='2'
c='0.0'

data=None
f='b3lyp_iao_b_'+detgen+'_N'+n+'_Ndet'+ndet+'_c'+c+'.pickle'
df=pd.read_pickle(f)

#1-body linear regression 
y=df['E']
y-=min(y)
Xr=df.drop(columns=['E','spin']+['nmo'+str(j) for j in range(14)])
#Xr=df[['nmo'+str(j) for j in range(5,14)]]

#Combined
ols=sm.OLS(y,Xr).fit()
print(ols.summary())
plt.plot(y,ols.predict(Xr),'go')
plt.plot(y,y,'r--')
plt.xlabel('E_B3LYP-E_0 (eV)')
plt.ylabel('E_PRED-E_0 (eV)')
plt.show()

#S=1/2
y=df['E'][df['spin']==1/2]
y-=min(y)
X=Xr[df['spin']==1/2]
ols=sm.OLS(y,X).fit()
print(ols.summary())
plt.plot(y,ols.predict(X),'go')
plt.plot(y,y,'r--')
plt.xlabel('E_B3LYP-E_0 (eV)')
plt.ylabel('E_PRED-E_0 (eV)')
plt.show()
#r=abs(y-ols.predict(X))
#print(r)

#S=3/2
y=df['E'][df['spin']==3/2]
y-=min(y)
X=Xr[df['spin']==3/2]
ols=sm.OLS(y,X).fit()
print(ols.summary())
plt.plot(y,ols.predict(X),'go')
plt.plot(y,y,'r--')
plt.xlabel('E_B3LYP-E_0 (eV)')
plt.ylabel('E_PRED-E_0 (eV)')
plt.show()
#r=abs(y-ols.predict(X))
#print(r)

'''
#Predict on other data set
detgen='a'
n='100'
ndet='2'
c='0.0'

data=None
f='b3lyp_iao_b_'+detgen+'_N'+n+'_Ndet'+ndet+'_c'+c+'.pickle'
df=pd.read_pickle(f)

#1-body linear regression 
y=df['E'][df['spin']==3/2]
y-=min(y)
X=df[df['spin']==3/2].drop(columns=['E','spin'])
plt.plot(y,ols.predict(X),'go')
plt.plot(y,y,'r--')
plt.xlabel('E_B3LYP-E_0 (eV)')
plt.ylabel('E_PRED-E_0 (eV)')
plt.show()
'''
