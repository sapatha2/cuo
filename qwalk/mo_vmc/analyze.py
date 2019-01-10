import pandas as pd 
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

df=pd.read_pickle("run1a/ex_a_Ndet10_gsw0.9_gosling.pickle")
y=df['energy']
X=df.drop(columns=['energy','energy_err','base_state'])
X['pi1']=X['y1']+X['x1']
X['pi2']=X['y2']+X['x2']
#X=X.drop(columns=['x1','y1','x2','y2','s1'])
X=sm.add_constant(X)

#Trace check 
plt.plot(X['pi1']+X['pi2']+X['s1']+X['s2']+X['s3'],'o')
plt.show()
exit(0)

ols=sm.OLS(y,X).fit()
print(ols.summary())
plt.plot(ols.predict(X),y,'o')
plt.plot(y,y,'--')
plt.show()
