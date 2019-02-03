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
  for basestate in range(8):
    for gsw in np.arange(0.1,1.01,0.1):
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/gosling.pickle' 
      small_df=pd.read_pickle(f)
      
      small_df['basestate']=basestate
      if(np.around(gsw,2)==1.0): small_df['basestate']=-1
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0)
  return df

def analyze(df):
  df['gsw']=np.round(df['gsw'],2)
  df['n_3d']=df['n_5']+df['n_6']+df['n_7']+df['n_8']+df['n_9']
  df['n_3dd']=df['n_8']+df['n_5']
  df['n_2ppi']=df['n_11']+df['n_12']
  df['n_2pz']=df['n_10']
  df['n_4s']=df['n_13']
  #PAIRPLOTS --------------------------------------------------------------------------
  #CORE
  #sns.pairplot(df,vars=['energy','n_0','n_1','n_2','n_3','n_4'],hue='basestate',markers=['o']+['.']*8)
  #D ORBITALS
  #sns.pairplot(df,vars=['energy','n_5','n_6','n_7','n_8','n_9'],hue='basestate',markers=['o']+['.']*8)
  #P and S ORBITALS
  #sns.pairplot(df,vars=['energy','n_10','n_11','n_12','n_13'],hue='basestate',markers=['o']+['.']*8)
  
  #ALL RELEVANT
  #sns.pairplot(df,vars=['energy','n_8','n_10','n_11','n_12','n_13'],hue='basestate',markers=['o']+['.']*8) 
  #plt.show()

  sns.pairplot(df,vars=['energy','n_3d','n_2pz','n_2ppi'],hue='basestate',markers=['o']+['.']*8)
  plt.show()

  #FITS --------------------------------------------------------------------------
  y=df['energy']
  yerr=df['energy_err']
  #X=df[['n_3d','n_2pz','3dz2-2pz','4s-2pz','U_4s']]
  X=df[['n_3d','n_2ppi','n_2pz']]
  X=sm.add_constant(X)
  #beta=2
  #for beta in np.arange(0,5,0.5):
  for beta in [0]:
    wls=sm.WLS(y,X,weights=np.exp(-beta*(y-min(y)))).fit()
    print(wls.summary())
    plt.plot(wls.predict(X),df['energy'],'o')
    plt.plot(df['energy'],df['energy'],'--k')
    plt.xlabel('Predicted (eV)')
    plt.ylabel('VMC (eV)')
    plt.show()

if __name__=='__main__':
  df=collectdf()
  analyze(df)
