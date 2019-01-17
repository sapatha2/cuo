import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.api as sm
import pandas as pd

f='run1s/ex_s_Ndet2_gsw0.0_gosling.pickle'
df=pd.read_pickle(f)
#sns.pairplot(df,vars=['energy','J_4s_3d','n_4s','n_3d','n_2ppi','n_2pz'],hue='base_state')
#plt.show()

X=df[['n_3d','n_2p']]
X=sm.add_constant(X)
y=df['energy']
pred=sm.OLS(y,X).fit().predict(X)
df['resid']=df['energy']-pred
sns.pairplot(df,vars=['resid','J_4s_3d','n_4s','n_3d','n_2p','3dpi-2ppi','3dz2-2pz','4s-2pz','4s-3dz2'],hue='base_state')
plt.show()
