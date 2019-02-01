import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV 
from sklearn.model_selection import cross_val_score

def collectdf():
  df=None
  for basestate in range(16):
    for gsw in np.arange(0.5,1.01,0.1):
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/gosling.pickle' 
      small_df=pd.read_pickle(f)
      
      small_df['basestate']=basestate
      if(np.around(gsw,2)==1.0): small_df['basestate']=-1
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0)
  return df

def analyze(df):
  df['gsw']=np.round(df['gsw'],2)
  #PAIRPLOTS --------------------------------------------------------------------------
  sns.pairplot(df,vars=['energy','n_3d','n_2ppi','n_2pz','U_4s','J_4s_3d'],hue='basestate',markers=['o']+['.']*16)
  plt.show()
  exit(0)

  #FITS --------------------------------------------------------------------------
  y=df['energy']
  yerr=df['energy_err']
  X=df[['n_3d','n_2ppi','n_2pz','3dz2-2pz','4s-2pz','3dpi-2ppi']]
  X=sm.add_constant(X)
  beta=2
  for beta in np.arange(0,5,0.5):
    wls=sm.WLS(y,X,weights=np.exp(-beta*(y-min(y)))).fit()
    print(wls.summary())
    df['pred']=wls.predict(X)
    sns.pairplot(df,vars=['energy','pred'],hue='basestate',markers=['o']+['.']*9)
    plt.show()

if __name__=='__main__':
  df=collectdf()
  analyze(df)
