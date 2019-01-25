import pandas as pd 
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np 

df=pd.read_pickle("run1a/ex_a_Ndet10_gsw0.9_gosling.pickle")
df=df[df['base_state']=='4SigmaM']
df['pi1']=df['y1']+df['x1']
df['pi2']=df['y2']+df['x2']
df['z']=df['s1']+df['s2']
df['spin']=df['sy1']+df['sx1']+df['ss1']
df['3d']=df['y1']+df['x1']+df['s1']
df['2p']=df['y2']+df['x2']+df['s2']
df['4s']=df['s3']

'''
print(np.mean(df['energy']),np.var(df['energy']))
print(np.mean(df['3d']),np.var(df['3d']))
print(np.mean(df['2p']),np.var(df['2p']))
print(np.mean(df['4s']),np.var(df['4s']))
'''

tr1=df['3d']+df['2p']+df['4s']
tr2=df['2p']+df['4s']
print(np.mean(tr1),np.var(tr1))
print(np.mean(tr2),np.var(tr2))

exit(0)
#sns.pairplot(df,vars=['energy','pi1','pi2','s1','s2','s3','spin'],hue='base_state')
#sns.pairplot(df,vars=['energy','3d','2p','4s','spin'])
#plt.show()

y=df['energy']
X=df[['pi1','pi2','s2','s3']]
#X=df[['3d','4s','s2']]
X=sm.add_constant(X)
ols=sm.OLS(y,X).fit()
print(ols.summary())
plt.plot(ols.predict(X),y,'.')
plt.show()
