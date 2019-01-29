import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import OrthogonalMatchingPursuit 
from sklearn.model_selection import cross_val_score

def collectdf():
  df=None
  for basestate in np.arange(4):
    for gsw in np.arange(0.1,1.0,0.1):
      if(basestate==0): f='gsw'+str(np.round(gsw,2))+'/gosling.pickle'
      else: f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/gosling.pickle' 
      small_df=pd.read_pickle(f)
      small_df['basestate']=basestate
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0)
  small_df=pd.read_pickle('base/base_gosling.pickle')
  small_df['gsw']=0.0
  small_df['basestate']=-1
  df = pd.concat((df,small_df),axis=0)
  return df

def analyze(df):
  df['gsw']=np.round(df['gsw'],2)
  #Full
  #sns.pairplot(df,vars=['energy','n_3d','J_4s_3d'],hue='basestate',markers=['o','.','.','.','.'])
  #plt.show()

  #Each basestate
  df=df[(df['basestate']==3)+(df['basestate']==-1)]
  sns.pairplot(df,vars=['energy','n_3d','J_4s_3d'],hue='gsw',markers=['o','.','.','.','.','.','.','.','.','.'])
  plt.show()
  
if __name__=='__main__':
  df=collectdf()
  analyze(df)
