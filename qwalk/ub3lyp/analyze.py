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
  for gsw in np.arange(0.1,1.0,0.1):
    f='gsw'+str(np.round(gsw,2))+'/gosling.pickle'
    small_df=pd.read_pickle(f)
    if(df is None): df = small_df
    else: df = pd.concat((df,small_df),axis=0)
  return df

def analyze(df):
  df['gsw']=np.round(df['gsw'],2)
  sns.pairplot(df,vars=['energy','n_3d','J_4s_3d'],hue='gsw',markers=['o','.','.','.','.','.','.','.','.','.'])
  plt.savefig('pairplot.pdf',bbox_inches='tight')
  exit(0)
  
  ###################
  #STAGE ONE
  ###################
  #1-body pairplot
  #sns.pairplot(df,vars=['energy','n_3d','n_2ppi','n_2pz','3dpi-2ppi','3dz2-2pz','4s-2pz','4s-3dz2'],hue='gsw')
  
  #U pairplot
  #sns.pairplot(df,vars=['energy','U_3d','U_2p','U_4s'],hue='gsw')
  
  #J pairplot
  #sns.pairplot(df,vars=['energy','J_4s_3d','J_4s_2p','J_3d_2p'],hue='gsw')

  #Full 
  #sns.pairplot(df,vars=['energy','n_3d','4s-3dz2','U_3d','U_4s','J_3d_2p','J_4s_2p','J_4s_3d'],hue='gsw')
  #plt.savefig('first_stage.pdf',bbox_inches='tight')

  '''
  y=df['energy']
  X=df[['n_3d','4s-3dz2','J_4s_3d']]
  X=sm.add_constant(X)
  ols=sm.OLS(y,X).fit()
  print(ols.summary())
  df['fit']=ols.predict(X)
  sns.pairplot(df,vars=['energy','fit'],hue='gsw')
  plt.show()
  '''
  
  ################
  #STAGE TWO 
  ################
  '''
  X=df.drop(columns=['energy','energy_err','gsw','spin'])
  y=df['energy']
  parms=[]
  r2_score=[]
  cv_score=[]
  for n in range(1,len(list(X))+1):
    reg=OrthogonalMatchingPursuit(n_nonzero_coefs=n).fit(X,y)
    parms.append(list(reg.coef_)+[reg.intercept_])
    r2_score.append(reg.score(X,y))
    cv_score.append(cross_val_score(reg,X,y,cv=5))
  parms=np.array(parms)
  r2_score=np.array(r2_score)
  cv_score=np.array(cv_score)
  #plt.plot(r2_score,'og')
  #plt.plot(cv_score,'or')
  #plt.show()
  
  ind=np.where(parms[1]!=0)[0][:-1] #Don't include coeff
  print(ind)
  print(np.array(list(X))[ind])
  print(parms[1][ind])
  '''

if __name__=='__main__':
  df=collectdf()
  analyze(df)
