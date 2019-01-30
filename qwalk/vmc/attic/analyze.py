import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import pandas as pd
import numpy as np 
sns.set(style="ticks")

'''
labels=np.array(["4s","3dxy","3dyz","3dz2","3dxz","3dx2y2","2px","2py","2pz"])
fname='run1s/s_Ndet10_gsw0.7_gosling.pickle'
df=pd.read_pickle(fname)
olabels=np.where('obdm' in list(df))[0]
print(olabels)
exit(0)
Ulabels=['tbdm_updown_%s_%s_%s_%s'%(s,s,s,s) for s in labels]
X=df[['energy']+Ulabels]
X['tbdm_updown_3dd_3dd_3dd_3dd']=X['tbdm_updown_3dxy_3dxy_3dxy_3dxy']+X['tbdm_updown_3dx2y2_3dx2y2_3dx2y2_3dx2y2']
X['tbdm_updown_3dpi_3dpi_3dpi_3dpi']=X['tbdm_updown_3dxz_3dxz_3dxz_3dxz']+X['tbdm_updown_3dyz_3dyz_3dyz_3dyz']
X['tbdm_updown_2ppi_2ppi_2ppi_2ppi']=X['tbdm_updown_2px_2px_2px_2px']+X['tbdm_updown_2py_2py_2py_2py']
X['tbdm_updown_d_d_d_d']=-1*(X['tbdm_updown_3dd_3dd_3dd_3dd']+X['tbdm_updown_3dpi_3dpi_3dpi_3dpi']+X['tbdm_updown_3dz2_3dz2_3dz2_3dz2'])
X=X.drop(columns=['tbdm_updown_3dxy_3dxy_3dxy_3dxy','tbdm_updown_3dx2y2_3dx2y2_3dx2y2_3dx2y2',
'tbdm_updown_3dxz_3dxz_3dxz_3dxz','tbdm_updown_3dyz_3dyz_3dyz_3dyz',
'tbdm_updown_2px_2px_2px_2px','tbdm_updown_2py_2py_2py_2py'])
#sns.pairplot(X)
#plt.show()

X['obdm_4s_4s']=df['obdm_up_4s_4s']+df['obdm_down_4s_4s']
X['obdm_2pz_2pz']=df['obdm_up_2pz_2pz']+df['obdm_down_2pz_2pz']
X['obdm_4s_2pz']=df['obdm_up_4s_2pz']+df['obdm_down_4s_2pz']
X['obdm_3dz2_2pz']=df['obdm_up_3dz2_2pz']+df['obdm_down_3dz2_2pz']
'''

'''
labels=np.array(list(df))
std=df.std()
ind=(std>3e-1)
print(labels[ind])
'''


fname='run1s/ex2_s_Ndet10_gsw0.7_gosling.pickle'
#fname='run2a/ex2_a_Ndet10_gsw0.8_gosling.pickle'
df=None
df=pd.read_pickle(fname)
df=df[(df['energy']-min(df['energy']))<3]
y=df['energy']
#X=df[['n_4s','n_2pz','n_2ppi','4s-2pz','3dz2-2pz','4s-3dz2','3dpi-2ppi']]
X=df[['n_3d']]
X=sm.add_constant(X)
#X=df.drop(columns=['energy','energy_err','gsw','detgen','obdm_4s_3dz2','obdm_3dz2_3dz2','obdm_3dpi_3dpi'])
#for zz in list(X):
#  if('tbdm' in zz): X=df.drop(columns=zz)
X=sm.add_constant(X)
beta=-3
wls=sm.WLS(y,X,weights=np.exp(beta*(y-min(y)))).fit()
print(wls.summary())

plt.ylabel('E_VMC (eV)')
plt.xlabel('E_Pred (eV)')
plt.errorbar(wls.predict(X),y,yerr=df['energy_err'],fmt='bo')
plt.plot(y,y,'g--')
plt.title('Singles space 1-body fit')
#plt.savefig('s_Ndet10_gsw0.7_df.pdf')
plt.show()
