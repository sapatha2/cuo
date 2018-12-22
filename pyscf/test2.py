import pandas as pd 
import numpy as np 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 

t=pd.read_pickle('test1.pickle')
t['E']+=-213.5799904*27.2114

t2=pd.read_pickle('test1b.pickle')
t2['E']+=-213.522801*27.2114

df=pd.concat((t,t2),axis=0)

y=df['E']
y-=min(y)
X=df.drop(columns=['E'])
ols=sm.OLS(y,X).fit()
print(ols.summary())
plt.plot(y,ols.predict(X),'o')
plt.plot(y,y)
plt.show()
