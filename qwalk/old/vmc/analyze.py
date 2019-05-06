import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import statsmodels.api as sm 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit
sns.set_palette('hls',7)

df=None
#for fin in ['run1s/ex_s_Ndet10_gsw0.7_gosling.pickle']:
for fin in ['run1a/ex_a_Ndet10_gsw0.8_gosling.pickle']:
#for fin in ['run1a/ex_a_Ndet10_gsw0.8_gosling.pickle','run1s/ex_s_Ndet10_gsw0.7_gosling.pickle']:
  new=pd.read_pickle(fin) 
  new['ex']=fin.split("/")[0]
  if(df is None): df=new
  else: df=pd.concat((df,new),axis=0)

df=df[df['base_state']=='2X']
#df=df[['energy','n_3dz2','n_3dd','n_3dpi','n_3d','n_2ppi','n_2pz','n_2p','n_4s']]
#df=df[['energy','3dz2-2pz','4s-2pz','4s-3dz2','3dpi-2ppi']]
#sns.pairplot(df)
#plt.show()
#exit(0)
print(list(df))
print(df.shape)

y=df['energy']
X=df[['n_4s','n_3d','n_2ppi','4s-2pz','3dz2-2pz','3dpi-2ppi','4s-3dz2']]
#X=df.drop(columns=['energy','energy_err','base_state','ex'])
#X=df[['V_3d']]
#X=df[['n_4s','n_2pz','n_2ppi']]
X=sm.add_constant(X)

beta=-3
wls=sm.WLS(y,X,weights=np.exp(beta*(y-min(y)))).fit()
print(wls.summary())
plt.errorbar(wls.predict(X),y,yerr=df['energy_err'],fmt='o')
plt.plot(y,y)
plt.show()
