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
    for gsw in np.arange(0.1,1.0,0.1):
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/gosling.pickle' 
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
  #PAIRPLOTS --------------------------------------------------------------------------
  #Full
  sns.pairplot(df,vars=['energy','n_3d','n_2ppi','n_2pz','U_4s'],hue='basestate',markers=['o']+['.']*9)
  #df=df[(df['basestate']==7)+(df['basestate']==-1)]
  #sns.pairplot(df,vars=['energy','n_3d','n_2ppi','n_2pz'],hue='basestate',markers=['o']+['.']*1)
  plt.show()
  exit(0)
  #plt.savefig('all_base.pdf',bbox_inches='tight')

  #Each basestate
  #for b in range(8):
  #  dat=df[(df['basestate']==b)+(df['basestate']==-1)]
  #  sns.pairplot(dat,vars=['energy','n_3d','J_4s_3d'],hue='gsw',markers=['o']+['.']*9)
  #  plt.savefig(str(b)+'_base.pdf',bbox_inches='tight')
  #  plt.close()
  
  #LINREG ---------------------------------------------------------------------------
  #Full
  #df=df[(df['basestate']==0)+(df['basestate']==-1)]
  '''
  y=df['energy']
  yerr=df['energy_err']
  X=df[['n_3d','n_2ppi','n_2pz','n_4s','3dz2-2pz','4s-3dz2','4s-2pz']]
  reg=OrthogonalMatchingPursuitCV(cv=5).fit(X,y)
  print(reg.score(X,y))
  print(reg.n_nonzero_coefs_,reg.n_iter_)
  print(reg.intercept_)
  print(reg.coef_[reg.coef_!=0])
  print(np.array(list(X))[reg.coef_!=0])
  #df['pred']=reg.predict(X)
  #sns.pairplot(df,vars=['energy','pred'],hue='basestate',markers=['o']+['.']*8)
  #sns.pairplot(df,vars=['energy','pred'],hue='gsw',markers=['o']+['.']*9)
  #plt.savefig('all_fit.pdf',bbox_inches='tight')
  '''
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

  #GROUND STATE ONLY
  '''
  0.9537834073367943
  5 5
  -5768.597210709244
  [-3.39486121 -2.06520991  1.21081355  1.88308435  1.62724933]
  ['n_3d' 'n_2pz' '3dpi-2ppi' '3dz2-2pz' '4s-2pz']
  '''
  #ALL SAMPLES
  '''
  0.9594805237781355
  5 5
  -5768.280708641785
  [-3.38004168 -2.25961628  1.19411177  2.19104091  1.75083078]
  ['n_3d' 'n_2pz' '3dpi-2ppi' '3dz2-2pz' '4s-2pz']
  '''

if __name__=='__main__':
  df=collectdf()
  analyze(df)
